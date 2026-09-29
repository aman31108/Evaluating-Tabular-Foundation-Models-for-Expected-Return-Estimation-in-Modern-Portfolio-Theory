import pandas as pd
from dataclasses import dataclass
from typing import Iterator

FEATURE_COLUMNS = [
    "ret_1m", "ret_3m", "ret_6m", "ret_12m",
    "vol_3m", "vol_12m", "avg_volume_3m",
]
TARGET_COLUMN = "target_next_month_return"


@dataclass
class WalkForwardSplit:
    
    split_id: int
    test_month: pd.Timestamp
    train_df: pd.DataFrame   
    test_df: pd.DataFrame   


def generate_walk_forward_splits(
    features_df: pd.DataFrame,
    min_train_months: int = 36,
) -> Iterator[WalkForwardSplit]:
    
    all_months = sorted(features_df["date"].unique())

    if len(all_months) <= min_train_months:
        raise ValueError(
            f"Not enough months of data ({len(all_months)}) for "
            f"min_train_months={min_train_months}. Reduce min_train_months "
            f"or provide more historical data."
        )

   
    eval_months = all_months[min_train_months:]

    for split_id, test_month in enumerate(eval_months):
        
        train_df = features_df[features_df["date"] < test_month].copy()
        
        test_df = features_df[features_df["date"] == test_month].copy()

        if train_df.empty or test_df.empty:
            continue

        yield WalkForwardSplit(
            split_id=split_id,
            test_month=test_month,
            train_df=train_df,
            test_df=test_df,
        )


def get_xy(df: pd.DataFrame):
    
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    return X, y


if __name__ == "__main__":
    
    import os
    from config import FEATURES_DIR, MIN_TRAIN_MONTHS

    features_path = os.path.join(FEATURES_DIR, "monthly_features.csv")
    df = pd.read_csv(features_path, parse_dates=["date"])

    splits = list(generate_walk_forward_splits(df, min_train_months=MIN_TRAIN_MONTHS))

    print(f"Generated {len(splits)} walk-forward splits\n")
    print("First split:")
    print(f"  test_month: {splits[0].test_month.date()}")
    print(f"  train rows: {len(splits[0].train_df)} "
          f"(dates {splits[0].train_df['date'].min().date()} to {splits[0].train_df['date'].max().date()})")
    print(f"  test rows:  {len(splits[0].test_df)} (should equal number of tickers)")

    print("\nLast split:")
    print(f"  test_month: {splits[-1].test_month.date()}")
    print(f"  train rows: {len(splits[-1].train_df)}")
    print(f"  test rows:  {len(splits[-1].test_df)}")

    
    leak_found = False
    for s in splits:
        if s.train_df["date"].max() >= s.test_month:
            leak_found = True
            print(f"  [LEAK DETECTED] split {s.split_id}: train includes test month!")
    if not leak_found:
        print("\n[OK] No temporal leakage detected across any split.")
