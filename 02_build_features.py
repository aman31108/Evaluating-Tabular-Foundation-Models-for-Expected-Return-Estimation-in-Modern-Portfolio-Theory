import os
import pandas as pd
import numpy as np

import sys
sys.path.append(os.path.dirname(__file__))
from config import TICKERS, DATA_DIR, FEATURES_DIR


def load_raw(ticker: str) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, "raw", f"{ticker}.csv")
    df = pd.read_csv(path, index_col="Date", parse_dates=True)
    return df


def build_monthly_features(ticker: str, daily_df: pd.DataFrame) -> pd.DataFrame:
    

    
    price_col = "Adj Close" if "Adj Close" in daily_df.columns else "Close"
    daily_close = daily_df[price_col].astype(float)

 
    monthly_close = daily_close.resample("ME").last()

    
    monthly_return = monthly_close.pct_change()

    
    daily_return = daily_close.pct_change()
    monthly_vol = daily_return.resample("ME").std()

    
    if "Volume" in daily_df.columns:
        monthly_volume = daily_df["Volume"].resample("ME").mean()
    else:
        monthly_volume = pd.Series(index=monthly_close.index, dtype=float)

    feat = pd.DataFrame(index=monthly_close.index)
    feat["ticker"] = ticker
    feat["price"] = monthly_close

    
    feat["ret_1m"] = monthly_return
    feat["ret_3m"] = monthly_close.pct_change(3)
    feat["ret_6m"] = monthly_close.pct_change(6)
    feat["ret_12m"] = monthly_close.pct_change(12)

    
    feat["vol_3m"] = monthly_vol.rolling(3).mean()
    feat["vol_12m"] = monthly_vol.rolling(12).mean()

   
    feat["avg_volume_3m"] = monthly_volume.rolling(3).mean()

   
    feat["target_next_month_return"] = monthly_return.shift(-1)

    return feat


def main():
    os.makedirs(FEATURES_DIR, exist_ok=True)

    all_features = []
    for ticker in TICKERS:
        print(f"  processing {ticker}...")
        daily_df = load_raw(ticker)
        feat = build_monthly_features(ticker, daily_df)
        all_features.append(feat)

    combined = pd.concat(all_features)
    combined = combined.reset_index().rename(columns={"Date": "date"})

    
    before = len(combined)
    combined = combined.dropna(subset=[
        "ret_12m", "vol_12m", "target_next_month_return"
    ])
    after = len(combined)
    print(f"\nDropped {before - after} rows with insufficient history / missing target")

    out_path = os.path.join(FEATURES_DIR, "monthly_features.csv")
    combined.to_csv(out_path, index=False)
    print(f"Saved {len(combined)} rows x {len(combined.columns)} columns to {out_path}")
    print(f"\nDate range: {combined['date'].min().date()} to {combined['date'].max().date()}")
    print(f"Tickers: {combined['ticker'].nunique()}")
    print(f"\nFirst few rows:")
    print(combined.head(3).to_string())


if __name__ == "__main__":
    main()
