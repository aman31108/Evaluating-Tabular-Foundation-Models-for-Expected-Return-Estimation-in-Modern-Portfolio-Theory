import os
import sys
import time
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

import sys as _sys
_sys.path.append(os.path.dirname(__file__))
from config import DATA_DIR, TICKERS

KRONOS_REPO_PATH = os.path.join(os.path.dirname(__file__), "..", "Kronos")
sys.path.append(KRONOS_REPO_PATH)


LOOKBACK_DAYS = 40
PRED_LEN_DAYS = 12
MODEL_NAME = "NeoQuasar/Kronos-small"
TOKENIZER_NAME = "NeoQuasar/Kronos-Tokenizer-base"

N_TEST_POINTS = 15  

def load_daily_prices(ticker: str) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, "raw", f"{ticker}.csv")
    df = pd.read_csv(path, index_col="Date", parse_dates=True)
    df = df.sort_index()
    rename_map = {"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}
    df = df.rename(columns=rename_map)
    df = df.reset_index().rename(columns={"Date": "timestamps"})
    return df[["timestamps", "open", "high", "low", "close", "volume"]]


def get_diagnostic_test_indices(daily_df: pd.DataFrame, n_points: int) -> list:

    n = len(daily_df)
    earliest_valid = LOOKBACK_DAYS
    latest_valid = n - PRED_LEN_DAYS - 1
    if latest_valid <= earliest_valid:
        raise ValueError("Not enough daily data for this lookback/horizon combination.")


    indices = np.linspace(earliest_valid, latest_valid, n_points, dtype=int)
    return sorted(set(indices.tolist()))


def run_diagnostic_for_ticker(ticker: str, predictor, test_indices: list) -> list:
    daily_df = load_daily_prices(ticker)
    rows = []

    for idx in test_indices:
        context = daily_df.iloc[idx - LOOKBACK_DAYS: idx].reset_index(drop=True)
        x_df = context[["open", "high", "low", "close", "volume"]]
        x_timestamp = context["timestamps"]

       
        forecast_start_date = daily_df["timestamps"].iloc[idx]
        assert x_timestamp.max() < forecast_start_date, "LEAK: context includes forecast start date or later"

        y_timestamp = daily_df["timestamps"].iloc[idx: idx + PRED_LEN_DAYS].reset_index(drop=True)

        pred_df = predictor.predict(
            df=x_df,
            x_timestamp=x_timestamp,
            y_timestamp=y_timestamp,
            pred_len=PRED_LEN_DAYS,
            T=0.6,      
            top_p=0.90,
            sample_count=1, 
        )

     
        last_real_close = x_df["close"].iloc[-1]
        predicted_end_close = pred_df["close"].iloc[-1]
        y_pred = (predicted_end_close - last_real_close) / last_real_close

        actual_end_close = daily_df["close"].iloc[idx + PRED_LEN_DAYS - 1]
        y_true = (actual_end_close - last_real_close) / last_real_close

        rows.append({
            "ticker": ticker,
            "test_idx": idx,
            "forecast_start_date": forecast_start_date,
            "y_true": y_true,
            "y_pred": y_pred,
        })

    return rows


def main():
    print("Loading Kronos model and tokenizer...")
    from model import Kronos, KronosTokenizer, KronosPredictor 

    tokenizer = KronosTokenizer.from_pretrained(TOKENIZER_NAME)
    model = Kronos.from_pretrained(MODEL_NAME)
    predictor = KronosPredictor(model, tokenizer, max_context=LOOKBACK_DAYS + 10)


    reference_df = load_daily_prices(TICKERS[0])
    test_indices = get_diagnostic_test_indices(reference_df, N_TEST_POINTS)
    print(f"Testing at {len(test_indices)} diagnostic points x {len(TICKERS)} tickers "
          f"= {len(test_indices) * len(TICKERS)} predictions\n")
    print(f"Settings: lookback={LOOKBACK_DAYS} days, horizon={PRED_LEN_DAYS} days, "
          f"T=0.6, top_p=0.90, sample_count=1\n")

    all_rows = []
    start_time = time.time()
    for i, ticker in enumerate(TICKERS):
        print(f"  [{i+1}/{len(TICKERS)}] {ticker}...")
        rows = run_diagnostic_for_ticker(ticker, predictor, test_indices)
        all_rows.extend(rows)
        elapsed = time.time() - start_time
        print(f"      elapsed so far: {elapsed/60:.1f} min")

    result_df = pd.DataFrame(all_rows)

    
    monthly_ic, monthly_rankic = [], []
    for date, group in result_df.groupby("forecast_start_date"):
        y_true = group["y_true"].values
        y_pred = group["y_pred"].values
        if len(y_true) < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
            continue
        ic, _ = pearsonr(y_pred, y_true)
        rankic, _ = spearmanr(y_pred, y_true)
        monthly_ic.append(ic)
        monthly_rankic.append(rankic)

    test_mae = (result_df["y_pred"] - result_df["y_true"]).abs().mean()

    print(f"\n=== DIAGNOSTIC RESULT (lookback=40, horizon=12, matching Kronos's own paper) ===")
    print(f"Predictions made: {len(result_df)}")
    print(f"Test MAE: {test_mae:.5f}")
    print(f"Avg IC: {np.mean(monthly_ic):.4f}")
    print(f"Avg RankIC: {np.mean(monthly_rankic):.4f}")
    print(f"\nCompare to:")
    print(f"  Main run (lookback=400, horizon=22): MAE=0.1199, RankIC=-0.0538")
    print(f"  Kronos's own reported (their full setup): RankIC=0.0675")

    os.makedirs("results", exist_ok=True)
    result_df.to_csv("results/kronos_diagnostic.csv", index=False)
    print(f"\nSaved raw diagnostic predictions to results/kronos_diagnostic.csv")


if __name__ == "__main__":
    main()
