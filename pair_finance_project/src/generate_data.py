"""
Step 2: Synthetic dataset generator.

Simulates a debtor population with behavioral/contact features and generates
a per-channel response outcome with realistic, intentionally-encoded patterns
(and noise), so the dataset is learnable but not trivially separable.

Domain assumptions encoded here are based on publicly stated patterns in the
debt-collection / fintech contact-optimization space (e.g. younger debtors
skew toward digital/SMS channels, higher debt amounts slow down response,
time-of-day affects open rates). This is 100% synthetic data -- no real
debtor information is used or represented.
"""

import numpy as np
import pandas as pd

RNG_SEED = 42
N_DEBTORS = 4000
CHANNELS = ["email", "sms", "post"]


def generate_debtor_profiles(n=N_DEBTORS, seed=RNG_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    age_bracket = rng.choice(
        ["18-25", "26-35", "36-50", "51-65", "65+"],
        size=n,
        p=[0.15, 0.30, 0.28, 0.18, 0.09],
    )

    # Debt amount: log-normal, skewed toward smaller debts
    debt_amount = np.round(rng.lognormal(mean=5.5, sigma=0.9, size=n), 2)
    debt_amount = np.clip(debt_amount, 20, 15000)

    # Days since last contact
    days_since_last_contact = rng.integers(0, 90, size=n)

    # Number of prior contact attempts (more attempts often = harder case)
    prior_contact_attempts = rng.poisson(lam=2.2, size=n)
    prior_contact_attempts = np.clip(prior_contact_attempts, 0, 15)

    # Time of day of last contact attempt (bucketed)
    time_of_day_last_contact = rng.choice(
        ["morning", "afternoon", "evening", "night"],
        size=n,
        p=[0.22, 0.33, 0.35, 0.10],
    )

    # Day of week of last contact
    day_of_week_last_contact = rng.choice(
        ["mon", "tue", "wed", "thu", "fri", "sat", "sun"], size=n
    )

    # Partial payment history: 0 = never, 1 = at least one partial payment made
    partial_payment_history = rng.choice([0, 1], size=n, p=[0.62, 0.38])

    # Historical per-channel response rate for this debtor's segment
    # (engineered later from age bracket, but store a noisy "true" propensity
    # per channel here to drive the label generation)
    age_email_bias = {
        "18-25": -0.3, "26-35": 0.1, "36-50": 0.4, "51-65": 0.3, "65+": 0.1
    }
    age_sms_bias = {
        "18-25": 0.7, "26-35": 0.5, "36-50": 0.0, "51-65": -0.3, "65+": -0.7
    }
    age_post_bias = {
        "18-25": -0.6, "26-35": -0.4, "36-50": -0.1, "51-65": 0.3, "65+": 0.8
    }

    df = pd.DataFrame({
        "debtor_id": [f"D{100000+i}" for i in range(n)],
        "age_bracket": age_bracket,
        "debt_amount": debt_amount,
        "days_since_last_contact": days_since_last_contact,
        "prior_contact_attempts": prior_contact_attempts,
        "time_of_day_last_contact": time_of_day_last_contact,
        "day_of_week_last_contact": day_of_week_last_contact,
        "partial_payment_history": partial_payment_history,
    })

    df["_age_email_bias"] = df["age_bracket"].map(age_email_bias)
    df["_age_sms_bias"] = df["age_bracket"].map(age_sms_bias)
    df["_age_post_bias"] = df["age_bracket"].map(age_post_bias)

    return df, rng


def _logistic(x):
    return 1 / (1 + np.exp(-x))


def generate_labels(df: pd.DataFrame, rng) -> pd.DataFrame:
    """
    Generate response probability per channel from a linear combination of
    features + bias terms + noise, then sample a binary outcome per channel.
    Also derive a single 'best_channel' and 'responded' label representing
    what actually happened (debtor is contacted via best predicted channel
    by the ops team's current heuristic, and either responds or not).
    """
    debt_z = (df["debt_amount"] - df["debt_amount"].mean()) / df["debt_amount"].std()
    recency_z = (df["days_since_last_contact"] - df["days_since_last_contact"].mean()) / df["days_since_last_contact"].std()
    attempts_z = (df["prior_contact_attempts"] - df["prior_contact_attempts"].mean()) / df["prior_contact_attempts"].std()

    time_bias = df["time_of_day_last_contact"].map(
        {"morning": 0.15, "afternoon": 0.25, "evening": 0.35, "night": -0.4}
    )
    weekend_bias = df["day_of_week_last_contact"].isin(["sat", "sun"]).map({True: -0.2, False: 0.05})

    payment_bias = df["partial_payment_history"].map({0: -0.15, 1: 0.35})

    base_noise_scale = 1.1  # keeps signal learnable but noisy, not trivial

    logit_email = (
        df["_age_email_bias"] + time_bias + weekend_bias + payment_bias
        - 0.35 * debt_z - 0.25 * recency_z - 0.20 * attempts_z
        + rng.normal(0, base_noise_scale, size=len(df))
    )
    logit_sms = (
        df["_age_sms_bias"] + 0.30 * time_bias + weekend_bias + payment_bias
        - 0.20 * debt_z - 0.30 * recency_z - 0.15 * attempts_z
        + rng.normal(0, base_noise_scale, size=len(df))
    )
    logit_post = (
        df["_age_post_bias"] - 0.1 * time_bias + payment_bias
        - 0.10 * debt_z - 0.10 * recency_z - 0.05 * attempts_z
        + rng.normal(0, base_noise_scale, size=len(df))
    )

    p_email = _logistic(logit_email)
    p_sms = _logistic(logit_sms)
    p_post = _logistic(logit_post)

    df["true_p_email"] = p_email
    df["true_p_sms"] = p_sms
    df["true_p_post"] = p_post

    df["responded_email"] = (rng.random(len(df)) < p_email).astype(int)
    df["responded_sms"] = (rng.random(len(df)) < p_sms).astype(int)
    df["responded_post"] = (rng.random(len(df)) < p_post).astype(int)

    df = df.drop(columns=["_age_email_bias", "_age_sms_bias", "_age_post_bias"])
    return df


def main():
    df, rng = generate_debtor_profiles()
    df = generate_labels(df, rng)
    out_path = "/home/claude/pair_finance_project/data/synthetic_debtors.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")
    print(df.head())
    print("\nResponse rates by channel:")
    print(df[["responded_email", "responded_sms", "responded_post"]].mean())


if __name__ == "__main__":
    main()
