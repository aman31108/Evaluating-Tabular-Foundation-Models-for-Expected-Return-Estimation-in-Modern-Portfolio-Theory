import os
import time
import numpy as np
import pandas as pd
from tabpfn import TabPFNRegressor

import sys
sys.path.append(os.path.dirname(__file__))
from config import FEATURES_DIR, PREDICTIONS_DIR, MIN_TRAIN_MONTHS
from walk_forward import generate_walk_forward_splits, get_xy


def run_tabpfn():
    features_path = os.path.join(FEATURES_DIR, "monthly_features.csv")
    df = pd.read_csv(features_path, parse_dates=["date"])

    rows = []
    splits = list(generate_walk_forward_splits(df, min_train_months=MIN_TRAIN_MONTHS))
    print(f"Running TabPFN-3 across {len(splits)} walk-forward splits...")
    print("(base tabular regressor, not the TS-3 time-series checkpoint)\n")

    start_time = time.time()

    for split in splits:
        X_train, y_train = get_xy(split.train_df)
        X_test, y_test = get_xy(split.test_df)

        model = TabPFNRegressor()
        model.fit(X_train, y_train)

       
        train_pred = model.predict(X_train)
        train_mae = np.abs(train_pred - y_train.values).mean()

       
        test_pred = model.predict(X_test)

        for i, (_, test_row) in enumerate(split.test_df.iterrows()):
            rows.append({
                "split_id": split.split_id,
                "test_month": split.test_month,
                "ticker": test_row["ticker"],
                "y_true": test_row["target_next_month_return"],
                "y_pred": test_pred[i],
                "train_mae": train_mae,
            })

        elapsed = time.time() - start_time
        avg_per_split = elapsed / (split.split_id + 1)
        remaining = avg_per_split * (len(splits) - split.split_id - 1)
        if split.split_id % 10 == 0:
            print(f"  split {split.split_id}/{len(splits)} done "
                  f"(test_month={split.test_month.date()}, train_mae={train_mae:.5f}) "
                  f"| elapsed={elapsed/60:.1f}min, est. remaining={remaining/60:.1f}min")

    result_df = pd.DataFrame(rows)

    test_mae = (result_df["y_pred"] - result_df["y_true"]).abs().mean()
    avg_train_mae = result_df["train_mae"].mean()

    print(f"\nTabPFN-3 model:")
    print(f"  Predictions made: {len(result_df)}")
    print(f"  Avg train MAE: {avg_train_mae:.5f}")
    print(f"  Overall test MAE: {test_mae:.5f}")
    print(f"  Train/Test error ratio: {test_mae / avg_train_mae:.2f}x")
    print(f"  Total time: {(time.time() - start_time)/60:.1f} minutes")

    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    out_path = os.path.join(PREDICTIONS_DIR, "tabpfn.csv")
    result_df.to_csv(out_path, index=False)
    print(f"\nSaved predictions to {out_path}")


if __name__ == "__main__":
    run_tabpfn()
