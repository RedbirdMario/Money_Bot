"""Auto-UI for strategy parameters based on ParamDescriptor."""

from __future__ import annotations

import streamlit as st

from money_bot.strategies.base import Strategy, ParamDescriptor, ParamType
from money_bot.strategies.registry import get_registry, list_strategy_names


def render_strategy_selector(key_prefix: str = "") -> str:
    """Render a strategy dropdown and return the selected strategy name."""
    names = list_strategy_names()
    labels = {name: get_registry()[name].description or name for name in names}
    selected = st.selectbox(
        "Strategy",
        names,
        format_func=lambda n: labels.get(n, n),
        key=f"{key_prefix}strategy_select",
    )
    return selected


def render_strategy_params(strategy_name: str, key_prefix: str = "") -> dict:
    """Render widgets for a strategy's PARAMS and return param dict."""
    registry = get_registry()
    strategy_cls = registry[strategy_name]
    params = {}

    for p in strategy_cls.PARAMS:
        widget_key = f"{key_prefix}{strategy_name}_{p.name}"
        val = _render_param(p, widget_key)
        params[p.name] = val

    return params


def _render_param(p: ParamDescriptor, key: str):
    """Render a single parameter widget based on its type."""
    if p.param_type == ParamType.INT:
        return st.slider(
            p.label,
            min_value=int(p.min_val) if p.min_val is not None else 1,
            max_value=int(p.max_val) if p.max_val is not None else 100,
            value=int(p.default),
            step=int(p.step) if p.step is not None else 1,
            help=p.tooltip,
            key=key,
        )
    elif p.param_type == ParamType.FLOAT:
        return st.slider(
            p.label,
            min_value=float(p.min_val) if p.min_val is not None else 0.0,
            max_value=float(p.max_val) if p.max_val is not None else 1.0,
            value=float(p.default),
            step=float(p.step) if p.step is not None else 0.01,
            help=p.tooltip,
            key=key,
        )
    elif p.param_type == ParamType.BOOL:
        return st.checkbox(
            p.label,
            value=bool(p.default),
            help=p.tooltip,
            key=key,
        )
    elif p.param_type == ParamType.SELECT:
        return st.selectbox(
            p.label,
            options=p.options,
            index=p.options.index(p.default) if p.default in p.options else 0,
            help=p.tooltip,
            key=key,
        )
    return p.default


def build_strategy_from_ui(strategy_name: str, params: dict) -> Strategy:
    """Instantiate a Strategy from registry name + param dict."""
    registry = get_registry()
    strategy_cls = registry[strategy_name]
    return strategy_cls.from_params(params)
