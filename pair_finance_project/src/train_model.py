"""
Steps 5-7: Core model, calibration, and decision layer.

- One gradient boosting classifier per channel (HistGradientBoostingClassifier,
  no external deps needed beyond sklearn -- easy to swap for XGBoost/LightGBM).
- Each model is wrapped in CalibratedClassifierCV (isotonic regression) so
  output probabilities are trustworthy, not just rank-ordered.
- Evaluated with AUROC + a reliability (calibration) diagram per channel.
- Decision layer turns calibrated probabilities into a channel recommendation
  with a confidence flag.
"""

import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.frozen import FrozenEstimator
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, brier_score_loss

TRAIN_PATH = "/home/claude/pair_finance_project/data/train.csv"
TEST_PATH = "/home/claude/pair_finance_project/data/test.csv"
PLOTS_DIR = "/home/claude/pair_finance_project/plots"
MODELS_DIR = "/home/claude/pair_finance_project/models"

CHANNELS = ["email", "sms", "post"]

# Confidence below this -> flag for human review instead of auto-acting
# (set from the observed confidence distribution -- see README for how this
# would be chosen in production: balancing auto-action coverage vs. error cost)
CONFIDENCE_THRESHOLD = 0.60


def get_feature_cols(df: pd.DataFrame):
    exclude = {
        "debtor_id", "true_p_email", "true_p_sms", "true_p_post",
        "responded_email", "responded_sms", "responded_post",
    }
    return [c for c in df.columns if c not in exclude]


def add_segment_response_rate(train: pd.DataFrame, test: pd.DataFrame):
    """
    Engineered feature: historical response rate for this debtor's age segment,
    computed ONLY on train to avoid leakage, then mapped onto test.
    Returns the rate map too, so it can be persisted for inference-time use.
    """
    age_cols = [c for c in train.columns if c.startswith("age_")]
    full_rate_map = {}
    for channel in CHANNELS:
        rate_map = {}
        for age_col in age_cols:
            mask = train[age_col] == 1
            rate_map[age_col] = train.loc[mask, f"responded_{channel}"].mean() if mask.sum() > 0 else train[f"responded_{channel}"].mean()
        for split_df in (train, test):
            seg_rate = np.zeros(len(split_df))
            for age_col in age_cols:
                seg_rate += split_df[age_col].values * rate_map[age_col]
            split_df[f"segment_rate_{channel}"] = seg_rate
        full_rate_map[channel] = {c.replace("age_", ""): v for c, v in rate_map.items()}
    return train, test, full_rate_map


def train_and_calibrate(train, test):
    feature_cols = get_feature_cols(train)
    results = {}
    models = {}

    for channel in CHANNELS:
        y_train_full = train[f"responded_{channel}"].values
        y_test = test[f"responded_{channel}"].values
        X_train_full = train[feature_cols].values
        X_test = test[feature_cols].values

        # Hold out a calibration slice from train (never seen by the base model's
        # fit) -- calibrating on the model's own training data would be circular
        # and would understate true miscalibration.
        X_fit, X_calib, y_fit, y_calib = train_test_split(
            X_train_full, y_train_full, test_size=0.3, random_state=42, stratify=y_train_full
        )

        raw_model = HistGradientBoostingClassifier(
            max_iter=150, max_depth=4, learning_rate=0.08, random_state=42
        )
        raw_model.fit(X_fit, y_fit)

        # Calibrate the frozen (already-fit) model on the held-out calibration slice
        calibrated = CalibratedClassifierCV(FrozenEstimator(raw_model), method="isotonic")
        calibrated.fit(X_calib, y_calib)

        p_calibrated = calibrated.predict_proba(X_test)[:, 1]
        p_raw = raw_model.predict_proba(X_test)[:, 1]

        auroc = roc_auc_score(y_test, p_calibrated)
        brier_calibrated = brier_score_loss(y_test, p_calibrated)
        brier_raw = brier_score_loss(y_test, p_raw)

        results[channel] = {
            "auroc": auroc,
            "brier_calibrated": brier_calibrated,
            "brier_raw": brier_raw,
            "p_calibrated": p_calibrated,
            "p_raw": p_raw,
            "y_test": y_test,
        }
        models[channel] = calibrated

        print(f"[{channel}] AUROC={auroc:.3f}  Brier(calibrated)={brier_calibrated:.4f}  "
              f"Brier(raw)={brier_raw:.4f}  (lower Brier = better)")

    return results, models, feature_cols


def plot_reliability_diagrams(results):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, channel in zip(axes, CHANNELS):
        r = results[channel]
        frac_pos_cal, mean_pred_cal = calibration_curve(r["y_test"], r["p_calibrated"], n_bins=8, strategy="quantile")
        frac_pos_raw, mean_pred_raw = calibration_curve(r["y_test"], r["p_raw"], n_bins=8, strategy="quantile")

        ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration", alpha=0.5)
        ax.plot(mean_pred_raw, frac_pos_raw, "o-", color="#C44E52", label="Raw model")
        ax.plot(mean_pred_cal, frac_pos_cal, "o-", color="#4C72B0", label="Calibrated (isotonic)")
        ax.set_title(f"{channel.upper()} reliability diagram")
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Observed response rate")
        ax.legend(fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/reliability_diagrams.png", dpi=150)
    plt.close()
    print(f"Saved reliability diagrams to {PLOTS_DIR}/reliability_diagrams.png")


def decision_layer(test: pd.DataFrame, results: dict, threshold=CONFIDENCE_THRESHOLD) -> pd.DataFrame:
    """
    Turn per-channel calibrated probabilities into a single recommendation:
    the highest-probability channel, plus a flag for whether confidence
    clears the auto-action threshold.
    """
    probs = np.column_stack([results[c]["p_calibrated"] for c in CHANNELS])
    best_idx = probs.argmax(axis=1)
    best_channel = np.array(CHANNELS)[best_idx]
    best_conf = probs.max(axis=1)

    decisions = pd.DataFrame({
        "debtor_id": test["debtor_id"].values,
        "recommended_channel": best_channel,
        "confidence": best_conf.round(3),
        "needs_human_review": best_conf < threshold,
    })
    decisions["action"] = np.where(
        decisions["needs_human_review"],
        "Flag for human review",
        "Auto-contact via " + decisions["recommended_channel"],
    )
    return decisions


def main():
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    train, test, segment_rate_map = add_segment_response_rate(train, test)

    results, models, feature_cols = train_and_calibrate(train, test)
    plot_reliability_diagrams(results)

    decisions = decision_layer(test, results)
    decisions.to_csv("/home/claude/pair_finance_project/data/decisions.csv", index=False)

    review_rate = decisions["needs_human_review"].mean()
    print(f"\n{review_rate:.1%} of test cases flagged for human review "
          f"(confidence < {CONFIDENCE_THRESHOLD})")
    print(decisions.head(10).to_string(index=False))

    # Persist models + feature columns for the demo app
    import os
    os.makedirs(MODELS_DIR, exist_ok=True)
    for channel, model in models.items():
        joblib.dump(model, f"{MODELS_DIR}/model_{channel}.joblib")
    with open(f"{MODELS_DIR}/feature_cols.json", "w") as f:
        json.dump(feature_cols, f)
    with open(f"{MODELS_DIR}/segment_rate_map.json", "w") as f:
        json.dump(segment_rate_map, f, indent=2)

    summary = {c: {"auroc": results[c]["auroc"], "brier_calibrated": results[c]["brier_calibrated"]} for c in CHANNELS}
    with open("/home/claude/pair_finance_project/data/model_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nModels and feature schema saved to", MODELS_DIR)


if __name__ == "__main__":
    main()
