import os
import time
import shutil
import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor

import sys
sys.path.append(os.path.dirname(__file__))
from config import FEATURES_DIR, PREDICTIONS_DIR, MIN_TRAIN_MONTHS
from walk_forward import generate_walk_forward_splits, FEATURE_COLUMNS, TARGET_COLUMN


AG_MODEL_DIR = os.path.join("_ag_tmp")  # AutoGluon needs a folder per fit; we reuse/clear this


def run_autogluon():
    features_path = os.path.join(FEATURES_DIR, "monthly_features.csv")
    df = pd.read_csv(features_path, parse_dates=["date"])

    rows = []
    splits = list(generate_walk_forward_splits(df, min_train_months=MIN_TRAIN_MONTHS))
    print(f"Running AutoGluon (good_quality) across {len(splits)} walk-forward splits...")
    print("This is slower than CatBoost/LightGBM — let it run, don't close the terminal.\n")

    start_time = time.time()

    for split in splits:
  
        train_data = split.train_df[FEATURE_COLUMNS + [TARGET_COLUMN]]
        test_data = split.test_df[FEATURE_COLUMNS + [TARGET_COLUMN]]

       
        if os.path.exists(AG_MODEL_DIR):
            shutil.rmtree(AG_MODEL_DIR)

        predictor = TabularPredictor(
            label=TARGET_COLUMN,
            problem_type="regression",
            path=AG_MODEL_DIR,
            verbosity=0,  
        )
        predictor.fit(
            train_data,
            presets="good_quality",
            time_limit=120,   
        )

        train_pred = predictor.predict(train_data[FEATURE_COLUMNS])
        train_mae = np.abs(train_pred.values - train_data[TARGET_COLUMN].values).mean()

        test_pred = predictor.predict(test_data[FEATURE_COLUMNS])

        for i, (_, test_row) in enumerate(split.test_df.iterrows()):
            rows.append({
                "split_id": split.split_id,
                "test_month": split.test_month,
                "ticker": test_row["ticker"],
                "y_true": test_row[TARGET_COLUMN],
                "y_pred": test_pred.values[i],
                "train_mae": train_mae,
            })

        elapsed = time.time() - start_time
        avg_per_split = elapsed / (split.split_id + 1)
        remaining = avg_per_split * (len(splits) - split.split_id - 1)
        print(f"  split {split.split_id}/{len(splits)} done "
              f"(test_month={split.test_month.date()}, train_mae={train_mae:.5f}) "
              f"| elapsed={elapsed/60:.1f}min, est. remaining={remaining/60:.1f}min")

    if os.path.exists(AG_MODEL_DIR):
        shutil.rmtree(AG_MODEL_DIR)

    result_df = pd.DataFrame(rows)

    test_mae = (result_df["y_pred"] - result_df["y_true"]).abs().mean()
    avg_train_mae = result_df["train_mae"].mean()

    print(f"\nAutoGluon model:")
    print(f"  Predictions made: {len(result_df)}")
    print(f"  Avg train MAE: {avg_train_mae:.5f}")
    print(f"  Overall test MAE: {test_mae:.5f}")
    print(f"  Train/Test error ratio: {test_mae / avg_train_mae:.2f}x")
    print(f"  Total time: {(time.time() - start_time)/60:.1f} minutes")

    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    out_path = os.path.join(PREDICTIONS_DIR, "autogluon.csv")
    result_df.to_csv(out_path, index=False)
    print(f"\nSaved predictions to {out_path}")


if __name__ == "__main__":
    run_autogluon()
