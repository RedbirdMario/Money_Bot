"""Composite strategy — combine multiple strategies via voting."""

from __future__ import annotations

from enum import Enum

import pandas as pd

from money_bot.types import Signal, SignalType, Side
from money_bot.strategies.base import Strategy


class VotingMode(Enum):
    UNANIMOUS = "unanimous"  # All must agree on direction
    MAJORITY = "majority"   # >50% agree
    ANY = "any"             # At least one signals


class _PositionState(Enum):
    FLAT = "flat"
    LONG = "long"
    SHORT = "short"


class CompositeStrategy(Strategy):
    """Combine multiple sub-strategies via voting.

    Each sub-strategy generates signals independently.
    Composite derives per-strategy position state over time,
    then applies voting rule to determine combined position.
    """

    name = "composite"
    description = "Composite — Combine multiple strategies with voting (unanimous, majority, any)."

    def __init__(
        self,
        sub_strategies: list[Strategy],
        voting_mode: VotingMode = VotingMode.MAJORITY,
    ):
        self.sub_strategies = sub_strategies
        self.voting_mode = voting_mode

    def get_params(self) -> dict:
        return {
            "voting_mode": self.voting_mode.value,
            "sub_strategies": [s.name for s in self.sub_strategies],
        }

    def generate_signals(self, df: pd.DataFrame) -> list[Signal]:
        n = len(self.sub_strategies)
        if n == 0:
            return []

        # 1. Collect all signals from sub-strategies
        all_sub_signals: list[list[Signal]] = []
        for strat in self.sub_strategies:
            all_sub_signals.append(strat.generate_signals(df))

        # 2. Build per-strategy position state at each timestamp
        # Collect all unique signal timestamps
        all_timestamps: set[pd.Timestamp] = set()
        for signals in all_sub_signals:
            for sig in signals:
                all_timestamps.add(sig.timestamp)

        if not all_timestamps:
            return []

        sorted_ts = sorted(all_timestamps)

        # For each sub-strategy, build a position timeline
        sub_states: list[dict[pd.Timestamp, _PositionState]] = []
        for signals in all_sub_signals:
            states: dict[pd.Timestamp, _PositionState] = {}
            current = _PositionState.FLAT
            sig_map: dict[pd.Timestamp, list[Signal]] = {}
            for sig in signals:
                sig_map.setdefault(sig.timestamp, []).append(sig)

            for ts in sorted_ts:
                if ts in sig_map:
                    for sig in sig_map[ts]:
                        if sig.signal_type == SignalType.ENTRY:
                            current = (_PositionState.LONG if sig.side == Side.LONG
                                       else _PositionState.SHORT)
                        elif sig.signal_type == SignalType.EXIT:
                            current = _PositionState.FLAT
                states[ts] = current
            sub_states.append(states)

        # 3. At each timestamp, count votes and apply voting rule
        composite_signals: list[Signal] = []
        composite_state = _PositionState.FLAT

        for ts in sorted_ts:
            long_votes = sum(1 for s in sub_states if s.get(ts, _PositionState.FLAT) == _PositionState.LONG)
            short_votes = sum(1 for s in sub_states if s.get(ts, _PositionState.FLAT) == _PositionState.SHORT)

            desired = self._apply_voting(long_votes, short_votes, n)

            if desired != composite_state:
                # Get price at this timestamp
                if ts in df.index:
                    price = df.loc[ts, "close"]
                else:
                    # Find nearest
                    idx = df.index.get_indexer([ts], method="nearest")[0]
                    price = df["close"].iloc[idx]

                # Emit exit if currently in position
                if composite_state != _PositionState.FLAT:
                    exit_side = Side.LONG if composite_state == _PositionState.LONG else Side.SHORT
                    composite_signals.append(
                        Signal(ts, SignalType.EXIT, exit_side, price, f"composite_vote_{desired.value}")
                    )

                # Emit entry if new desired is not flat
                if desired != _PositionState.FLAT:
                    entry_side = Side.LONG if desired == _PositionState.LONG else Side.SHORT
                    composite_signals.append(
                        Signal(ts, SignalType.ENTRY, entry_side, price, f"composite_vote_{desired.value}")
                    )

                composite_state = desired

        return composite_signals

    def _apply_voting(
        self, long_votes: int, short_votes: int, total: int
    ) -> _PositionState:
        if self.voting_mode == VotingMode.UNANIMOUS:
            if long_votes == total:
                return _PositionState.LONG
            if short_votes == total:
                return _PositionState.SHORT
            return _PositionState.FLAT

        elif self.voting_mode == VotingMode.MAJORITY:
            threshold = total / 2
            if long_votes > threshold:
                return _PositionState.LONG
            if short_votes > threshold:
                return _PositionState.SHORT
            return _PositionState.FLAT

        elif self.voting_mode == VotingMode.ANY:
            # Prefer long if both signal (arbitrary)
            if long_votes > 0 and long_votes >= short_votes:
                return _PositionState.LONG
            if short_votes > 0:
                return _PositionState.SHORT
            return _PositionState.FLAT

        return _PositionState.FLAT
