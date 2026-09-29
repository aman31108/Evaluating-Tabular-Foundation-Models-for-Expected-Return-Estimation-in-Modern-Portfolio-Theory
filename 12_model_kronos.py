import os
import sys
import time
import numpy as np
import pandas as pd

import sys as _sys
_sys.path.append(os.path.dirname(__file__))
from config import DATA_DIR, PREDICTIONS_DIR, TICKERS, MIN_TRAIN_MONTHS

KRONOS_REPO_PATH = os.path.join(os.path.dirname(__file__), "..", "Kronos")
sys.path.append(KRONOS_REPO_PATH)

LOOKBACK_DAYS = 400          
PRED_LEN_DAYS = 22           
MODEL_NAME = "NeoQuasar/Kronos-small"          
TOKENIZER_NAME = "NeoQuasar/Kronos-Tokenizer-base"


def load_daily_prices(ticker: str) -> pd.DataFrame:
    """Load one ticker's raw daily data and rename columns to what Kronos expects."""
    path = os.path.join(DATA_DIR, "raw", f"{ticker}.csv")
    df = pd.read_csv(path, index_col="Date", parse_dates=True)
    df = df.sort_index()
    rename_map = {"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}
    df = df.rename(columns=rename_map)

    df = df.reset_index().rename(columns={"Date": "timestamps"})
    return df[["timestamps", "open", "high", "low", "close", "volume"]]


def get_target_test_months() -> list:
    
    reference_path = os.path.join(PREDICTIONS_DIR, "historical_mean.csv")
    if not os.path.exists(reference_path):
        raise FileNotFoundError(
           
        )
    ref_df = pd.read_csv(reference_path, parse_dates=["test_month"])
    return sorted(ref_df["test_month"].unique())


def run_kronos_for_ticker(ticker: str, predictor, test_months: list) -> list:
    daily_df = load_daily_prices(ticker)

    rows = []
    for split_id, test_month in enumerate(test_months):
       
        history = daily_df[daily_df["timestamps"] < test_month.replace(day=1)]
        if len(history) < 30:
            continue  
        context = history.tail(LOOKBACK_DAYS).reset_index(drop=True)
        x_df = context[["open", "high", "low", "close", "volume"]]
        x_timestamp = context["timestamps"]

       
        y_timestamp = pd.bdate_range(
            start=x_timestamp.iloc[-1] + pd.Timedelta(days=1),
            periods=PRED_LEN_DAYS,
        )
        y_timestamp = pd.Series(y_timestamp)

        pred_df = predictor.predict(
            df=x_df,
            x_timestamp=x_timestamp,
            y_timestamp=y_timestamp,
            pred_len=PRED_LEN_DAYS,
            T=1.0,
            top_p=0.9,
            sample_count=1,
        )

        
        last_real_close = x_df["close"].iloc[-1]
        predicted_month_end_close = pred_df["close"].iloc[-1]
        y_pred = (predicted_month_end_close - last_real_close) / last_real_close

        
        actual_month_data = daily_df[
            (daily_df["timestamps"] >= test_month.replace(day=1)) &
            (daily_df["timestamps"] <= test_month)
        ]
        if len(actual_month_data) == 0:
            continue
        actual_month_end_close = actual_month_data["close"].iloc[-1]
        y_true = (actual_month_end_close - last_real_close) / last_real_close

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
    print("Loading Kronos model and tokenizer (first run downloads weights from HuggingFace)...")
    from model import Kronos, KronosTokenizer, KronosPredictor 

    tokenizer = KronosTokenizer.from_pretrained(TOKENIZER_NAME)
    model = Kronos.from_pretrained(MODEL_NAME)
    predictor = KronosPredictor(model, tokenizer, max_context=LOOKBACK_DAYS + 20)

    test_months = get_target_test_months()
    print(f"Aligned to {len(test_months)} test months (matching the other 5 models): "
          f"{test_months[0].date()} to {test_months[-1].date()}\n")

    all_rows = []
    start_time = time.time()
    for i, ticker in enumerate(TICKERS):
        print(f"  [{i+1}/{len(TICKERS)}] {ticker}...")
        rows = run_kronos_for_ticker(ticker, predictor, test_months)
        all_rows.extend(rows)
        elapsed = time.time() - start_time
        print(f"      {len(rows)} predictions, elapsed {elapsed/60:.1f} min")

    result_df = pd.DataFrame(all_rows)

    test_mae = (result_df["y_pred"] - result_df["y_true"]).abs().mean()
    print(f"\nKronos (supplementary, raw-OHLCV input):")
    print(f"  Predictions made: {len(result_df)}")
    print(f"  Test MAE: {test_mae:.5f}")
    print(f"  (train_mae not applicable - zero-shot model, no training step)")

    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    out_path = os.path.join(PREDICTIONS_DIR, "kronos.csv")
    result_df.to_csv(out_path, index=False)
    print(f"\nSaved predictions to {out_path}")


if __name__ == "__main__":
    main()
