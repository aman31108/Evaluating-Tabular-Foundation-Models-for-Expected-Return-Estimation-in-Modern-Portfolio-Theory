import os
import numpy as np
import pandas as pd

import sys
sys.path.append(os.path.dirname(__file__))
from config import FEATURES_DIR, PREDICTIONS_DIR, RESULTS_DIR, TICKERS

MODELS = ["historical_mean", "catboost", "lightgbm", "autogluon", "tabpfn", "kronos"]
COV_LOOKBACK_MONTHS = 36
TRANSACTION_COST_BPS = 15 


def load_realized_returns_wide() -> pd.DataFrame:
   
    features_path = os.path.join(FEATURES_DIR, "monthly_features.csv")
    df = pd.read_csv(features_path, parse_dates=["date"])
    wide = df.pivot(index="date", columns="ticker", values="ret_1m")
    return wide.sort_index()


def load_all_predictions() -> dict:
    
    preds = {}
    for name in MODELS:
        path = os.path.join(PREDICTIONS_DIR, f"{name}.csv")
        df = pd.read_csv(path, parse_dates=["test_month"])
        preds[name] = df
    return preds


def tangency_weights(mu: np.ndarray, cov: np.ndarray) -> np.ndarray:
  
    n = len(mu)
    try:
        inv_cov = np.linalg.pinv(cov)  
        raw_w = inv_cov @ mu
    except np.linalg.LinAlgError:
        return np.full(n, 1.0 / n)

    raw_w = np.clip(raw_w, 0, None) 
    total = raw_w.sum()

    if total <= 1e-8:  
        return np.full(n, 1.0 / n)

    return raw_w / total


def compute_portfolio_series(model_name: str, preds_df: pd.DataFrame,
                              returns_wide: pd.DataFrame) -> pd.DataFrame:
    
    rows = []
    prev_weights = np.full(len(TICKERS), 1.0 / len(TICKERS))  

    test_months = sorted(preds_df["test_month"].unique())

    for test_month in test_months:
        month_preds = preds_df[preds_df["test_month"] == test_month].set_index("ticker")
        month_preds = month_preds.reindex(TICKERS) 

        
        history = returns_wide[returns_wide.index < test_month].tail(COV_LOOKBACK_MONTHS)
        history = history[TICKERS] 
        cov = history.cov().values

        if model_name == "equal_weight":
            weights = np.full(len(TICKERS), 1.0 / len(TICKERS))
        else:
            mu = month_preds["y_pred"].values
            weights = tangency_weights(mu, cov)

        
        turnover = 0.5 * np.abs(weights - prev_weights).sum()

      
        y_true = month_preds["y_true"].values
        gross_return = float(np.dot(weights, y_true))

        cost = (TRANSACTION_COST_BPS / 10000.0) * turnover
        net_return = gross_return - cost

        rows.append({
            "model": model_name,
            "test_month": test_month,
            "gross_return": gross_return,
            "net_return": net_return,
            "turnover": turnover,
        })

        prev_weights = weights

    return pd.DataFrame(rows)


def summarize_portfolio(returns: pd.Series, label: str) -> dict:
    """Compute Annual Return, Sharpe, Sortino, Max Drawdown for one return series."""
    r = returns.values
    n_months = len(r)

   
    compounded_growth = np.prod(1 + r)
    annual_return = compounded_growth ** (12 / n_months) - 1

   
    sharpe = (r.mean() / r.std()) * np.sqrt(12) if r.std() > 0 else np.nan

    
    downside = r[r < 0]
    downside_std = downside.std() if len(downside) > 1 else np.nan
    sortino = (r.mean() / downside_std) * np.sqrt(12) if downside_std and downside_std > 0 else np.nan


    wealth = np.cumprod(1 + r)
    running_max = np.maximum.accumulate(wealth)
    drawdown = (wealth - running_max) / running_max
    max_drawdown = drawdown.min()

    return {
        f"{label}_annual_return": annual_return,
        f"{label}_sharpe": sharpe,
        f"{label}_sortino": sortino,
        f"{label}_max_drawdown": max_drawdown,
    }


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Loading realized returns (for covariance estimation)...")
    returns_wide = load_realized_returns_wide()

    print("Loading all 5 models' predictions...")
    preds = load_all_predictions()

    print("\nBuilding portfolios (this runs 6 portfolios x 83 months)...")
    all_monthly = []

    for model_name in MODELS:
        print(f"  {model_name}...")
        monthly_df = compute_portfolio_series(model_name, preds[model_name], returns_wide)
        all_monthly.append(monthly_df)

    
    print("  equal_weight (1/N baseline)...")
    monthly_df = compute_portfolio_series("equal_weight", preds[MODELS[0]], returns_wide)
    all_monthly.append(monthly_df)

    monthly_all = pd.concat(all_monthly, ignore_index=True)

    monthly_path = os.path.join(RESULTS_DIR, "portfolio_monthly_returns.csv")
    monthly_all.to_csv(monthly_path, index=False)
    print(f"\nSaved monthly returns to {monthly_path}")

   
    summary_rows = []
    for model_name in MODELS + ["equal_weight"]:
        sub = monthly_all[monthly_all["model"] == model_name]

        gross_metrics = summarize_portfolio(sub["gross_return"], "gross")
        net_metrics = summarize_portfolio(sub["net_return"], "net")
        avg_turnover = sub["turnover"].mean()

        row = {"model": model_name, "avg_turnover": avg_turnover}
        row.update(gross_metrics)
        row.update(net_metrics)
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(RESULTS_DIR, "portfolio_metrics.csv")
    summary_df.to_csv(summary_path, index=False)

    print(f"Saved portfolio summary to {summary_path}\n")
    display_cols = ["model", "gross_annual_return", "gross_sharpe", "gross_max_drawdown",
                     "avg_turnover", "net_annual_return", "net_sharpe"]
    print(summary_df[display_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
