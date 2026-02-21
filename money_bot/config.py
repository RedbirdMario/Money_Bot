"""Configuration classes for backtesting and Monte Carlo."""

from __future__ import annotations

from dataclasses import dataclass, field

from money_bot.types import TradingMode


@dataclass
class BacktestConfig:
    initial_capital: float = 10_000.0
    fee_rate: float = 0.001  # 0.1% per trade (taker)
    slippage_pct: float = 0.0005  # 0.05%
    position_size_pct: float = 1.0  # 100% of capital per trade
    max_drawdown_pct: float = 0.25  # -25% shutdown
    train_test_split: float = 0.7  # 70% train, 30% test
    use_train_test: bool = False
    stop_loss_pct: float | None = None  # e.g. 0.05 = 5%
    take_profit_pct: float | None = None  # e.g. 0.10 = 10%
    trading_mode: TradingMode = TradingMode.BOTH


@dataclass
class MonteCarloConfig:
    n_simulations: int = 1000
    confidence_levels: list[float] = field(
        default_factory=lambda: [0.05, 0.25, 0.50, 0.75, 0.95]
    )
    methods: list[str] = field(
        default_factory=lambda: ["shuffle", "bootstrap", "noise"]
    )
    noise_std: float = 0.1  # 10% of trade PnL std
    random_seed: int | None = 42
