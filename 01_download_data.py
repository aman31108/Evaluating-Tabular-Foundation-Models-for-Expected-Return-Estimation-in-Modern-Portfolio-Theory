import os
import time
import yfinance as yf
import pandas as pd

import sys
sys.path.append(os.path.dirname(__file__))
from config import TICKERS, START_DATE, END_DATE, DATA_DIR


def download_ticker(ticker: str, out_dir: str) -> bool:
    """Download one ticker's history and save to CSV. Returns True on success."""
    out_path = os.path.join(out_dir, f"{ticker}.csv")

    if os.path.exists(out_path):
        print(f"  [skip] {ticker} already downloaded")
        return True

    try:
        df = yf.download(
            ticker,
            start=START_DATE,
            end=END_DATE,
            auto_adjust=False,   # keep both Close and Adj Close separately
            progress=False,
        )

        if df is None or df.empty:
            print(f"  [FAIL] {ticker}: no data returned")
            return False

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.index.name = "Date"
        df.to_csv(out_path)
        print(f"  [ok]   {ticker}: {len(df)} rows, {df.index.min().date()} to {df.index.max().date()}")
        return True

    except Exception as e:
        print(f"  [FAIL] {ticker}: {e}")
        return False


def main():
    raw_dir = os.path.join(DATA_DIR, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    print(f"Downloading {len(TICKERS)} tickers from {START_DATE} to {END_DATE}")
    print(f"Saving to: {raw_dir}/\n")

    results = {}
    for ticker in TICKERS:
        results[ticker] = download_ticker(ticker, raw_dir)
        time.sleep(0.5)  # be polite to Yahoo Finance's servers

    ok = sum(results.values())
    print(f"\nDone: {ok}/{len(TICKERS)} tickers downloaded successfully")

    failed = [t for t, success in results.items() if not success]
    if failed:
        print(f"Failed tickers (re-run this script to retry): {failed}")


if __name__ == "__main__":
    main()
