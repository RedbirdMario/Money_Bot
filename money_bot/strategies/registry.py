"""Strategy registry — auto-discovery via @register decorator."""

from __future__ import annotations

from typing import Type

from money_bot.strategies.base import Strategy

_REGISTRY: dict[str, Type[Strategy]] = {}


def register(cls: Type[Strategy]) -> Type[Strategy]:
    """Decorator: register a Strategy subclass by its .name attribute."""
    _REGISTRY[cls.name] = cls
    return cls


def get_registry() -> dict[str, Type[Strategy]]:
    """Return the full registry dict (name -> class)."""
    _ensure_loaded()
    return dict(_REGISTRY)


def get_strategy_class(name: str) -> Type[Strategy]:
    """Lookup a strategy class by name. Raises KeyError if not found."""
    _ensure_loaded()
    return _REGISTRY[name]


def list_strategy_names() -> list[str]:
    """Return sorted list of registered strategy names."""
    _ensure_loaded()
    return sorted(_REGISTRY.keys())


_loaded = False


def _ensure_loaded():
    """Import all strategy modules once so @register decorators fire."""
    global _loaded
    if _loaded:
        return
    _loaded = True
    # Import every strategy module — order doesn't matter
    import money_bot.strategies.double_ema  # noqa: F401
    import money_bot.strategies.rsi_mean_reversion  # noqa: F401
    import money_bot.strategies.bollinger_breakout  # noqa: F401
    import money_bot.strategies.macd_histogram  # noqa: F401
    import money_bot.strategies.donchian_breakout  # noqa: F401
    import money_bot.strategies.stochastic_ema  # noqa: F401
    import money_bot.strategies.vwap_volume  # noqa: F401
