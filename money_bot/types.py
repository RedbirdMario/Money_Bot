"""Core data types for Money_Bot."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import pandas as pd


class Side(Enum):
    LONG = "long"
    SHORT = "short"


class TradingMode(Enum):
    BOTH = "both"
    LONG_ONLY = "long_only"
    SHORT_ONLY = "short_only"


class SignalType(Enum):
    ENTRY = "entry"
    EXIT = "exit"


@dataclass
class Signal:
    timestamp: pd.Timestamp
    signal_type: SignalType
    side: Side
    price: float
    reason: str = ""


@dataclass
class Trade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    side: Side
    entry_price: float
    exit_price: float
    size: float
    pnl: float
    pnl_pct: float
    fee_paid: float
    reason_entry: str = ""
    reason_exit: str = ""

    @property
    def duration(self) -> pd.Timedelta:
        return self.exit_time - self.entry_time

    @property
    def is_winner(self) -> bool:
        return self.pnl > 0


@dataclass
class BacktestResult:
    trades: list[Trade]
    equity_curve: pd.Series
    signals: list[Signal]
    metrics: dict[str, float]
    config: Any = None
    strategy_name: str = ""
    dataset_name: str = ""


@dataclass
class MonteCarloResult:
    original_metrics: dict[str, float]
    simulated_equity_curves: list[pd.Series]
    simulated_metrics: list[dict[str, float]]
    confidence_intervals: dict[str, dict[str, float]]
    ruin_probability: float
    curve_fit_score: float
    n_simulations: int = 0
    method: str = ""
