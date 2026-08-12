"""
Step 4: Feature engineering.
Encodes categoricals, builds engineered features, and produces a time-based
train/test split (using days_since_last_contact as a proxy timeline: debtors
contacted more recently are treated as the "later" cohort).
"""

import pandas as pd
import numpy as np

DATA_PATH = "/home/claude/pair_finance_project/data/synthetic_debtors.csv"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # One-hot encode categoricals
    df = pd.get_dummies(
        df,
        columns=["age_bracket", "time_of_day_last_contact", "day_of_week_last_contact"],
        prefix=["age", "tod", "dow"],
    )

    # Engineered feature: is_weekend
    dow_cols = [c for c in df.columns if c.startswith("dow_")]
    weekend_cols = [c for c in dow_cols if c in ("dow_sat", "dow_sun")]
    df["is_weekend_contact"] = df[weekend_cols].sum(axis=1) if weekend_cols else 0

    # Engineered feature: segment historical response rate (leave-one-out-safe
    # approximation using age bracket dummy columns reconstructed from originals)
    # We recompute segment rates on TRAIN split only later, inside train_model.py,
    # to avoid leakage. Here we just keep debt_amount log-transformed and a
    # normalized recency/attempts feature.
    df["log_debt_amount"] = np.log1p(df["debt_amount"])
    df["contact_intensity"] = df["prior_contact_attempts"] / (df["days_since_last_contact"] + 1)

    return df


def time_based_split(df: pd.DataFrame, test_frac=0.2):
    """
    Simulate a time-based split: treat debtors with LOWER days_since_last_contact
    as more recent (later in the simulated timeline) and hold those out as test set,
    which is more realistic than a random split for a system that will be deployed
    forward in time.
    """
    df_sorted = df.sort_values("days_since_last_contact", ascending=False).reset_index(drop=True)
    split_idx = int(len(df_sorted) * (1 - test_frac))
    train = df_sorted.iloc[:split_idx].reset_index(drop=True)
    test = df_sorted.iloc[split_idx:].reset_index(drop=True)
    return train, test


def main():
    df = pd.read_csv(DATA_PATH)
    df_feat = engineer_features(df)
    train, test = time_based_split(df_feat)

    train.to_csv("/home/claude/pair_finance_project/data/train.csv", index=False)
    test.to_csv("/home/claude/pair_finance_project/data/test.csv", index=False)

    print(f"Train: {len(train)} rows, Test: {len(test)} rows")
    print(f"Feature columns: {df_feat.shape[1]}")
    print(train.columns.tolist())


if __name__ == "__main__":
    main()
