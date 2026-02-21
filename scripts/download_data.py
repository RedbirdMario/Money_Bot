"""Download data utility script."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from money_bot.data.loader import load_data


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/download_data.py bybit:SOLUSDT:1h [limit]")
        print("       python scripts/download_data.py binance:BTCUSDT:4h 2000")
        sys.exit(1)

    source = sys.argv[1]
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 1000

    print(f"Downloading {source} (limit={limit})...")
    df = load_data(source, limit=limit)
    print(f"Got {len(df)} bars")
    print(f"Range: {df.index[0]} to {df.index[-1]}")
    print(df.tail())


if __name__ == "__main__":
    main()
