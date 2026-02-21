"""Abstract base class for vectorized strategies + ParamDescriptor system."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import pandas as pd

from money_bot.types import Signal


class ParamType(Enum):
    INT = "int"
    FLOAT = "float"
    BOOL = "bool"
    SELECT = "select"


@dataclass
class ParamDescriptor:
    name: str  # __init__ kwarg name
    label: str  # UI label
    param_type: ParamType
    default: Any
    min_val: float | None = None
    max_val: float | None = None
    step: float | None = None
    options: list[str] = field(default_factory=list)  # for SELECT
    tooltip: str = ""


class Strategy(ABC):
    """Base strategy — vectorized approach.

    Receives the full DataFrame, returns a list of Signals.
    Subclasses declare PARAMS for auto-UI and description for display.
    """

    name: str = "base"
    description: str = ""
    PARAMS: list[ParamDescriptor] = []

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> list[Signal]:
        """Generate entry/exit signals from OHLCV data."""
        ...

    def get_params(self) -> dict:
        """Return current strategy parameters."""
        return {}

    @classmethod
    def from_params(cls, params: dict) -> "Strategy":
        """Create strategy instance from a parameter dict."""
        # Filter to only known param names
        known = {p.name for p in cls.PARAMS}
        filtered = {k: v for k, v in params.items() if k in known}
        return cls(**filtered)
