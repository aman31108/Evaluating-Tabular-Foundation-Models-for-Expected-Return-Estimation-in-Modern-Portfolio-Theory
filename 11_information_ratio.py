import os
import numpy as np
import pandas as pd

import sys
sys.path.append(os.path.dirname(__file__))
from config import RESULTS_DIR

MODELS = ["historical_mean", "catboost", "lightgbm", "autogluon", "tabpfn", "kronos"]
BENCHMARK = "equal_weight"


def compute_ir(model_returns: np.ndarray, benchmark_returns: np.ndarray) -> tuple:
    """Returns (IR, mean active return, tracking error)."""
    active = model_returns - benchmark_returns
    mean_active = active.mean()
    tracking_error = active.std()
    ir = mean_active / tracking_error if tracking_error > 0 else np.nan
    
    ir_annualized = ir * np.sqrt(12)
    return ir_annualized, mean_active * 12, tracking_error * np.sqrt(12)


def main():
    monthly_path = os.path.join(RESULTS_DIR, "portfolio_monthly_returns.csv")
    monthly = pd.read_csv(monthly_path, parse_dates=["test_month"])

    
    gross_wide = monthly.pivot(index="test_month", columns="model", values="gross_return")
    net_wide = monthly.pivot(index="test_month", columns="model", values="net_return")

    if BENCHMARK not in gross_wide.columns:
        raise ValueError(f"Benchmark '{BENCHMARK}' not found in portfolio_monthly_returns.csv columns: {list(gross_wide.columns)}")

    rows = []
    for model in MODELS:
        if model not in gross_wide.columns:
            print(f"  [skip] {model}: not found in portfolio_monthly_returns.csv")
            continue

        ir_gross, active_ret_gross, te_gross = compute_ir(
            gross_wide[model].values, gross_wide[BENCHMARK].values
        )
        ir_net, active_ret_net, te_net = compute_ir(
            net_wide[model].values, net_wide[BENCHMARK].values
        )

        rows.append({
            "model": model,
            "gross_IR": ir_gross,
            "gross_active_return_annualized": active_ret_gross,
            "gross_tracking_error_annualized": te_gross,
            "net_IR": ir_net,
            "net_active_return_annualized": active_ret_net,
            "net_tracking_error_annualized": te_net,
        })
        print(f"  [ok]   {model}")

    result_df = pd.DataFrame(rows).sort_values("net_IR", ascending=False)

    out_path = os.path.join(RESULTS_DIR, "information_ratio.csv")
    result_df.to_csv(out_path, index=False)

    print(f"\nSaved to {out_path}\n")
    print(f"Benchmark: Equal-Weight (1/N) portfolio\n")
    display_cols = ["model", "gross_IR", "gross_active_return_annualized", "net_IR", "net_active_return_annualized"]
    print(result_df[display_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nIR > 0 means the model beat the 1/N benchmark on average.")
    print("IR measures CONSISTENCY of beating the benchmark, not just the size of the edge -")
    print("a small, steady edge (like Manager B in the lecture's example) gives a higher IR")
    print("than a large but erratic one (like Manager A).")


if __name__ == "__main__":
    main()
