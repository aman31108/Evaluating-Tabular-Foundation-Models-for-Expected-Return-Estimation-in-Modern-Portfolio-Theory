import os
import sys
import time
import numpy as np
import pandas as pd

import sys as _sys
_sys.path.append(os.path.dirname(__file__))
from config import DATA_DIR, PREDICTIONS_DIR, TICKERS

KRONOS_REPO_PATH = os.path.join(os.path.dirname(__file__), "..", "Kronos")
sys.path.append(KRONOS_REPO_PATH)


LOOKBACK_DAYS = 40
PRED_LEN_DAYS = 12
TEMPERATURE = 0.6
TOP_P = 0.90
SAMPLE_COUNT = 10  
MODEL_NAME = "NeoQuasar/Kronos-small"
TOKENIZER_NAME = "NeoQuasar/Kronos-Tokenizer-base"


def load_daily_prices(ticker: str) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, "raw", f"{ticker}.csv")
    df = pd.read_csv(path, index_col="Date", parse_dates=True)
    df = df.sort_index()
    rename_map = {"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}
    df = df.rename(columns=rename_map)
    df = df.reset_index().rename(columns={"Date": "timestamps"})
    return df[["timestamps", "open", "high", "low", "close", "volume"]]


def get_target_test_months() -> list:
    """Same 83 months as the other 5 models, read from historical_mean.csv."""
    reference_path = os.path.join(PREDICTIONS_DIR, "historical_mean.csv")
    if not os.path.exists(reference_path):
        raise FileNotFoundError("predictions/historical_mean.csv not found - run the core models first.")
    ref_df = pd.read_csv(reference_path, parse_dates=["test_month"])
    return sorted(ref_df["test_month"].unique())


def find_anchor_index(daily_df: pd.DataFrame, test_month) -> int:
   
    month_start = test_month.replace(day=1)
    candidates = daily_df.index[daily_df["timestamps"] >= month_start]
    if len(candidates) == 0:
        return None
    return candidates[0]


def run_for_ticker(ticker: str, predictor, test_months: list) -> list:
    daily_df = load_daily_prices(ticker)
    rows = []

    for split_id, test_month in enumerate(test_months):
        idx = find_anchor_index(daily_df, test_month)
        if idx is None or idx < LOOKBACK_DAYS or idx + PRED_LEN_DAYS > len(daily_df):
            continue  
        context = daily_df.iloc[idx - LOOKBACK_DAYS: idx].reset_index(drop=True)
        x_df = context[["open", "high", "low", "close", "volume"]]
        x_timestamp = context["timestamps"]

        forecast_start_date = daily_df["timestamps"].iloc[idx]
        assert x_timestamp.max() < forecast_start_date, "LEAK: context reaches forecast start or later"

        y_timestamp = daily_df["timestamps"].iloc[idx: idx + PRED_LEN_DAYS].reset_index(drop=True)

        pred_df = predictor.predict(
            df=x_df,
            x_timestamp=x_timestamp,
            y_timestamp=y_timestamp,
            pred_len=PRED_LEN_DAYS,
            T=TEMPERATURE,
            top_p=TOP_P,
            sample_count=SAMPLE_COUNT,
        )

        last_real_close = x_df["close"].iloc[-1]
        predicted_end_close = pred_df["close"].iloc[-1]
        y_pred = (predicted_end_close - last_real_close) / last_real_close

        actual_end_close = daily_df["close"].iloc[idx + PRED_LEN_DAYS - 1]
        y_true = (actual_end_close - last_real_close) / last_real_close

        rows.append({
            "split_id": split_id,
            "test_month": test_month,
            "ticker": ticker,
            "y_true": y_true,
            "y_pred": y_pred,
            "train_mae": np.nan,  
        })

    return rows


def main():
    print("Loading Kronos model and tokenizer...")
    from model import Kronos, KronosTokenizer, KronosPredictor  

    tokenizer = KronosTokenizer.from_pretrained(TOKENIZER_NAME)
    model = Kronos.from_pretrained(MODEL_NAME)
    predictor = KronosPredictor(model, tokenizer, max_context=LOOKBACK_DAYS + 10)

    test_months = get_target_test_months()
    print(f"Aligned to {len(test_months)} test months (same as the other 5 models): "
          f"{test_months[0].date()} to {test_months[-1].date()}")
    print(f"Settings: lookback={LOOKBACK_DAYS}d, horizon={PRED_LEN_DAYS}d, "
          f"T={TEMPERATURE}, top_p={TOP_P}, sample_count={SAMPLE_COUNT}\n")

    all_rows = []
    start_time = time.time()
    for i, ticker in enumerate(TICKERS):
        print(f"  [{i+1}/{len(TICKERS)}] {ticker}...")
        rows = run_for_ticker(ticker, predictor, test_months)
        all_rows.extend(rows)
        elapsed = time.time() - start_time
        print(f"      {len(rows)} predictions, elapsed {elapsed/60:.1f} min")

    result_df = pd.DataFrame(all_rows)
    test_mae = (result_df["y_pred"] - result_df["y_true"]).abs().mean()

    print(f"\n=== FULL RUN RESULT (lookback=40, horizon=12, matching Kronos's own paper) ===")
    print(f"Predictions made: {len(result_df)}")
    print(f"Test MAE: {test_mae:.5f}")
    print(f"\nRun src/10_ic_rankic.py on this file (add 'kronos_paper_settings' to MODELS)")
    print(f"to get Avg IC / Avg RankIC for full comparison.")

    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    out_path = os.path.join(PREDICTIONS_DIR, "kronos_paper_settings.csv")
    result_df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
