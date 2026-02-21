"""Standalone runner to start the Bookmap addon.

Usage::

    python -m money_bot.integrations.bookmap_runner --strategy double_ema --interval 3600
    python -m money_bot.integrations.bookmap_runner --strategy rsi_mean_reversion --params '{"rsi_period": 14}'
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from money_bot.integrations.bookmap_addon import BookmapAddon


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Money_Bot Bookmap Addon — draw strategy signals on the heatmap",
    )
    parser.add_argument(
        "--strategy",
        default="double_ema",
        help="Strategy name from the registry (default: double_ema)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=3600,
        help="Candle interval in seconds (default: 3600 = 1h)",
    )
    parser.add_argument(
        "--max-candles",
        type=int,
        default=500,
        help="Max candles to keep in rolling buffer (default: 500)",
    )
    parser.add_argument(
        "--params",
        default="{}",
        help='Strategy parameters as JSON string (default: "{}")',
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    try:
        params = json.loads(args.params)
    except json.JSONDecodeError as exc:
        print(f"Error: Invalid JSON in --params: {exc}", file=sys.stderr)
        sys.exit(1)

    addon = BookmapAddon(
        strategy_name=args.strategy,
        strategy_params=params,
        interval_seconds=args.interval,
        max_candles=args.max_candles,
    )

    logging.getLogger(__name__).info(
        "Starting Bookmap addon: strategy=%s interval=%ds",
        args.strategy,
        args.interval,
    )
    addon.start()


if __name__ == "__main__":
    main()
