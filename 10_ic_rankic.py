
import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

import sys
sys.path.append(os.path.dirname(__file__))
from config import PREDICTIONS_DIR, RESULTS_DIR

MODELS = ["historical_mean", "catboost", "lightgbm", "autogluon", "tabpfn","kronos","kronos_paper_settings"]


def compute_ic_rankic_for_model(name: str) -> dict:
    path = os.path.join(PREDICTIONS_DIR, f"{name}.csv")
    df = pd.read_csv(path, parse_dates=["test_month"])

    monthly_ic = []
    monthly_rankic = []

    for test_month, group in df.groupby("test_month"):
        y_true = group["y_true"].values
        y_pred = group["y_pred"].values

       
        if len(y_true) < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
            continue

        
        ic, _ = pearsonr(y_pred, y_true)
        monthly_ic.append(ic)

       
        rankic, _ = spearmanr(y_pred, y_true)
        monthly_rankic.append(rankic)

    avg_ic = np.mean(monthly_ic)
    avg_rankic = np.mean(monthly_rankic)
    std_ic = np.std(monthly_ic)
    std_rankic = np.std(monthly_rankic)

    
    hit_rate = np.mean(np.array(monthly_rankic) > 0)

    return {
        "model": name,
        "n_months": len(monthly_ic),
        "avg_IC": avg_ic,
        "avg_RankIC": avg_rankic,
        "std_IC": std_ic,
        "std_RankIC": std_rankic,
        "RankIC_positive_rate": hit_rate,
    }


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    rows = []
    for name in MODELS:
        path = os.path.join(PREDICTIONS_DIR, f"{name}.csv")
        if not os.path.exists(path):
            print(f"  [skip] {name}: predictions/{name}.csv not found")
            continue
        rows.append(compute_ic_rankic_for_model(name))
        print(f"  [ok]   {name}")

    result_df = pd.DataFrame(rows)
    result_df = result_df.sort_values("avg_RankIC", ascending=False)

    out_path = os.path.join(RESULTS_DIR, "ic_rankic.csv")
    result_df.to_csv(out_path, index=False)

    print(f"\nSaved to {out_path}\n")
    print(result_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nNote: RankIC values are typically small in real markets (Kronos itself")
    print("reports Avg. RankIC around 0.02-0.07) - this is normal, not a bug.")
    print("What matters is the RELATIVE ranking of models, not the absolute size.")


if __name__ == "__main__":
    main()
