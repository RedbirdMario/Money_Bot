"""Portfolio tracking — positions, equity curve, fees/slippage, SL/TP."""

from __future__ import annotations

import pandas as pd
import numpy as np

from money_bot.types import Signal, SignalType, Side, Trade, TradingMode
from money_bot.config import BacktestConfig


class Portfolio:
    """Tracks positions, calculates PnL, builds equity curve."""

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.initial_capital = config.initial_capital
        self.capital = config.initial_capital
        self.position: dict | None = None  # {side, entry_price, size, entry_time, reason}
        self.trades: list[Trade] = []
        self.equity_history: list[tuple[pd.Timestamp, float]] = []

    @property
    def peak_equity(self) -> float:
        if not self.equity_history:
            return self.initial_capital
        return max(eq for _, eq in self.equity_history)

    @property
    def current_drawdown_pct(self) -> float:
        peak = self.peak_equity
        if peak == 0:
            return 0.0
        return (peak - self.capital) / peak

    @property
    def max_drawdown_exceeded(self) -> bool:
        return self.current_drawdown_pct >= self.config.max_drawdown_pct

    def process_signals(
        self, signals: list[Signal], prices: pd.DataFrame
    ) -> list[Trade]:
        """Process signals chronologically, managing positions and equity.

        Applies trading mode filter and checks SL/TP between signals.
        """
        sorted_signals = sorted(signals, key=lambda s: s.timestamp)

        for i, signal in enumerate(sorted_signals):
            self._record_equity(signal.timestamp)

            if self.max_drawdown_exceeded:
                if self.position is not None:
                    self._close_position(signal.timestamp, signal.price, "max_dd_shutdown")
                break

            if signal.signal_type == SignalType.ENTRY and self.position is None:
                # Trading mode filter
                if self._should_skip_entry(signal.side):
                    continue
                self._open_position(signal)

                # Check SL/TP on bars between this signal and the next
                next_ts = (sorted_signals[i + 1].timestamp
                           if i + 1 < len(sorted_signals) else None)
                self._check_sl_tp(prices, signal.timestamp, next_ts)

            elif signal.signal_type == SignalType.EXIT and self.position is not None:
                self._close_position(signal.timestamp, signal.price, signal.reason)

        # Record final equity
        if self.equity_history:
            last_ts = self.equity_history[-1][0]
        else:
            last_ts = pd.Timestamp.now(tz="UTC")
        self._record_equity(last_ts)

        return self.trades

    def _should_skip_entry(self, side: Side) -> bool:
        mode = self.config.trading_mode
        if mode == TradingMode.LONG_ONLY and side == Side.SHORT:
            return True
        if mode == TradingMode.SHORT_ONLY and side == Side.LONG:
            return True
        return False

    def _check_sl_tp(
        self,
        prices: pd.DataFrame,
        after_ts: pd.Timestamp,
        before_ts: pd.Timestamp | None,
    ):
        """Check stop-loss and take-profit on bars between two signal timestamps."""
        if self.position is None:
            return
        sl = self.config.stop_loss_pct
        tp = self.config.take_profit_pct
        if sl is None and tp is None:
            return
        if prices.empty:
            return

        entry_price = self.position["entry_price"]
        side = self.position["side"]

        # Compute SL/TP price levels
        if side == Side.LONG:
            sl_price = entry_price * (1 - sl) if sl else None
            tp_price = entry_price * (1 + tp) if tp else None
        else:
            sl_price = entry_price * (1 + sl) if sl else None
            tp_price = entry_price * (1 - tp) if tp else None

        # Get bars in range (after_ts, before_ts]
        if before_ts is not None:
            mask = (prices.index > after_ts) & (prices.index <= before_ts)
        else:
            mask = prices.index > after_ts
        bars = prices.loc[mask]

        for idx in range(len(bars)):
            bar_low = bars["low"].iloc[idx]
            bar_high = bars["high"].iloc[idx]
            ts = bars.index[idx]

            # SL has priority over TP (conservative)
            if side == Side.LONG:
                if sl_price is not None and bar_low <= sl_price:
                    self._close_position(ts, sl_price, "stop_loss")
                    return
                if tp_price is not None and bar_high >= tp_price:
                    self._close_position(ts, tp_price, "take_profit")
                    return
            else:  # SHORT
                if sl_price is not None and bar_high >= sl_price:
                    self._close_position(ts, sl_price, "stop_loss")
                    return
                if tp_price is not None and bar_low <= tp_price:
                    self._close_position(ts, tp_price, "take_profit")
                    return

    def _open_position(self, signal: Signal):
        slippage = signal.price * self.config.slippage_pct
        if signal.side == Side.LONG:
            fill_price = signal.price + slippage
        else:
            fill_price = signal.price - slippage

        position_value = self.capital * self.config.position_size_pct
        fee = position_value * self.config.fee_rate
        size = (position_value - fee) / fill_price

        self.capital -= fee
        self.position = {
            "side": signal.side,
            "entry_price": fill_price,
            "size": size,
            "entry_time": signal.timestamp,
            "reason": signal.reason,
            "entry_fee": fee,
        }

    def _close_position(self, timestamp: pd.Timestamp, price: float, reason: str):
        pos = self.position
        slippage = price * self.config.slippage_pct

        if pos["side"] == Side.LONG:
            fill_price = price - slippage
            raw_pnl = (fill_price - pos["entry_price"]) * pos["size"]
        else:
            fill_price = price + slippage
            raw_pnl = (pos["entry_price"] - fill_price) * pos["size"]

        exit_fee = fill_price * pos["size"] * self.config.fee_rate
        net_pnl = raw_pnl - exit_fee
        total_fees = pos["entry_fee"] + exit_fee

        entry_value = pos["entry_price"] * pos["size"]
        pnl_pct = net_pnl / entry_value if entry_value > 0 else 0.0

        self.capital += entry_value + net_pnl

        trade = Trade(
            entry_time=pos["entry_time"],
            exit_time=timestamp,
            side=pos["side"],
            entry_price=pos["entry_price"],
            exit_price=fill_price,
            size=pos["size"],
            pnl=net_pnl,
            pnl_pct=pnl_pct,
            fee_paid=total_fees,
            reason_entry=pos["reason"],
            reason_exit=reason,
        )
        self.trades.append(trade)
        self.position = None

    def _record_equity(self, timestamp: pd.Timestamp):
        self.equity_history.append((timestamp, self.capital))

    def get_equity_curve(self) -> pd.Series:
        if not self.equity_history:
            return pd.Series(dtype=float)
        times, values = zip(*self.equity_history)
        return pd.Series(values, index=pd.DatetimeIndex(times), name="equity")
