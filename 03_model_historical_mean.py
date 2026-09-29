import os
import numpy as np
import pandas as pd

import sys
sys.path.append(os.path.dirname(__file__))
from config import FEATURES_DIR, PREDICTIONS_DIR, MIN_TRAIN_MONTHS
from walk_forward import generate_walk_forward_splits, TARGET_COLUMN


def run_historical_mean():
    features_path = os.path.join(FEATURES_DIR, "monthly_features.csv")
    df = pd.read_csv(features_path, parse_dates=["date"])

    rows = []
    for split in generate_walk_forward_splits(df, min_train_months=MIN_TRAIN_MONTHS):

    
        ticker_means = split.train_df.groupby("ticker")["ret_1m"].mean()

        
        train_pred = split.train_df["ticker"].map(ticker_means)
        train_true = split.train_df[TARGET_COLUMN]
        train_mae = (train_pred - train_true).abs().mean()

        
        for _, test_row in split.test_df.iterrows():
            ticker = test_row["ticker"]
            y_pred = ticker_means.get(ticker, np.nan)
            y_true = test_row[TARGET_COLUMN]

            rows.append({
                "split_id": split.split_id,
                "test_month": split.test_month,
                "ticker": ticker,
                "y_true": y_true,
                "y_pred": y_pred,
                "train_mae": train_mae,
            })

    result_df = pd.DataFrame(rows)


    test_mae = (result_df["y_pred"] - result_df["y_true"]).abs().mean()
    avg_train_mae = result_df["train_mae"].mean()

    print(f"Historical Mean model:")
    print(f"  Predictions made: {len(result_df)}")
    print(f"  Avg train MAE: {avg_train_mae:.5f}")
    print(f"  Overall test MAE: {test_mae:.5f}")
    print(f"  Train/Test error ratio: {test_mae / avg_train_mae:.2f}x")

    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    out_path = os.path.join(PREDICTIONS_DIR, "historical_mean.csv")
    result_df.to_csv(out_path, index=False)
    print(f"\nSaved predictions to {out_path}")


if __name__ == "__main__":
    run_historical_mean()
