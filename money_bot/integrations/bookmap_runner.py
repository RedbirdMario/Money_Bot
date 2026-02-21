"""Bookmap Python Addon runner for Money_Bot.

Bookmap launches this script with a socket port as sys.argv[1].
Configuration is done via environment variables or a JSON config file.

Usage — register this script path in Bookmap's Python API addon manager:
    /Users/marionakowitz/Projects/Money_Bot/money_bot/integrations/bookmap_runner.py

Configure via environment variables:
    export MB_STRATEGY=double_ema
    export MB_INTERVAL=60
    export MB_MAX_CANDLES=500
    export MB_PARAMS='{"fast_period": 20}'
    export MB_LOG_LEVEL=DEBUG

Or via config file (auto-detected next to this script):
    money_bot/integrations/bookmap_config.json
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

from money_bot.integrations.bookmap_addon import BookmapAddon

# Config file path (same directory as this script)
CONFIG_FILE = Path(__file__).parent / "bookmap_config.json"


def _load_config() -> dict:
    """Load config from JSON file, then override with environment variables."""
    config = {
        "strategy": "double_ema",
        "interval": 3600,
        "max_candles": 500,
        "params": {},
        "log_level": "INFO",
    }

    # Load from config file if it exists
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE) as f:
                file_config = json.load(f)
            config.update(file_config)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"Warning: Could not read {CONFIG_FILE}: {exc}", file=sys.stderr)

    # Environment variables override file config
    if os.environ.get("MB_STRATEGY"):
        config["strategy"] = os.environ["MB_STRATEGY"]
    if os.environ.get("MB_INTERVAL"):
        config["interval"] = int(os.environ["MB_INTERVAL"])
    if os.environ.get("MB_MAX_CANDLES"):
        config["max_candles"] = int(os.environ["MB_MAX_CANDLES"])
    if os.environ.get("MB_PARAMS"):
        config["params"] = json.loads(os.environ["MB_PARAMS"])
    if os.environ.get("MB_LOG_LEVEL"):
        config["log_level"] = os.environ["MB_LOG_LEVEL"]

    return config


def main() -> None:
    config = _load_config()

    logging.basicConfig(
        level=getattr(logging, config["log_level"]),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    logger = logging.getLogger(__name__)

    logger.info(
        "Starting Bookmap addon: strategy=%s interval=%ds max_candles=%d",
        config["strategy"],
        config["interval"],
        config["max_candles"],
    )

    addon = BookmapAddon(
        strategy_name=config["strategy"],
        strategy_params=config["params"],
        interval_seconds=config["interval"],
        max_candles=config["max_candles"],
    )
    addon.start()


if __name__ == "__main__":
    main()
