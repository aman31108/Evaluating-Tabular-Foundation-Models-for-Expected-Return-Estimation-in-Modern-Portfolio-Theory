import os
import numpy as np
import pandas as pd

import sys
sys.path.append(os.path.dirname(__file__))
from config import PREDICTIONS_DIR, RESULTS_DIR

MODELS = ["historical_mean", "catboost", "lightgbm", "autogluon", "tabpfn"]


def compute_metrics_for_model(name: str) -> dict:
    path = os.path.join(PREDICTIONS_DIR, f"{name}.csv")
    df = pd.read_csv(path)

    y_true = df["y_true"].values
    y_pred = df["y_pred"].values
    errors = y_pred - y_true

    mae = np.abs(errors).mean()
    rmse = np.sqrt((errors ** 2).mean())

    ss_res = (errors ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan


    valid = (y_true != 0) & (y_pred != 0)
    same_sign = np.sign(y_true[valid]) == np.sign(y_pred[valid])
    directional_accuracy = same_sign.mean() if valid.sum() > 0 else np.nan

    train_mae = df["train_mae"].mean()
    train_test_ratio = mae / train_mae if train_mae > 0 else np.nan

    return {
        "model": name,
        "n_predictions": len(df),
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "directional_accuracy": directional_accuracy,
        "train_mae": train_mae,
        "train_test_ratio": train_test_ratio,
    }


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    rows = []
    for name in MODELS:
        path = os.path.join(PREDICTIONS_DIR, f"{name}.csv")
        if not os.path.exists(path):
            print(f"  [skip] {name}: predictions/{name}.csv not found yet")
            continue
        rows.append(compute_metrics_for_model(name))
        print(f"  [ok]   {name}")

    result_df = pd.DataFrame(rows)

    out_path = os.path.join(RESULTS_DIR, "prediction_metrics.csv")
    result_df.to_csv(out_path, index=False)

    print(f"\nSaved to {out_path}\n")
    print(result_df.to_string(index=False, float_format=lambda x: f"{x:.5f}"))


if __name__ == "__main__":
    main()
