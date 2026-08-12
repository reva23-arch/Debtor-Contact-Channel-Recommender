"""
Step 8: Demo app.
Input a debtor profile -> output recommended contact channel + confidence
+ human-review flag. Run with: streamlit run app.py
"""

import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODELS_DIR = "/home/claude/pair_finance_project/models"
CHANNELS = ["email", "sms", "post"]
CONFIDENCE_THRESHOLD = 0.60

AGE_BRACKETS = ["18-25", "26-35", "36-50", "51-65", "65+"]
TIME_BUCKETS = ["morning", "afternoon", "evening", "night"]
DAYS_OF_WEEK = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]

@st.cache_resource
def load_models():
    models = {ch: joblib.load(f"{MODELS_DIR}/model_{ch}.joblib") for ch in CHANNELS}
    with open(f"{MODELS_DIR}/feature_cols.json") as f:
        feature_cols = json.load(f)
    with open(f"{MODELS_DIR}/segment_rate_map.json") as f:
        # stored as {channel: {age_bracket: rate}} -> re-key to {age_bracket: {channel: rate}}
        raw_map = json.load(f)
    segment_rates = {age: {} for age in AGE_BRACKETS}
    for ch, age_rates in raw_map.items():
        for age, rate in age_rates.items():
            segment_rates[age][ch] = rate
    return models, feature_cols, segment_rates


def build_feature_row(profile: dict, feature_cols: list, segment_rates: dict) -> pd.DataFrame:
    row = {c: 0 for c in feature_cols}
    row["debt_amount"] = profile["debt_amount"]
    row["days_since_last_contact"] = profile["days_since_last_contact"]
    row["prior_contact_attempts"] = profile["prior_contact_attempts"]
    row["partial_payment_history"] = profile["partial_payment_history"]
    row["log_debt_amount"] = np.log1p(profile["debt_amount"])
    row["contact_intensity"] = profile["prior_contact_attempts"] / (profile["days_since_last_contact"] + 1)

    age_col = f"age_{profile['age_bracket']}"
    if age_col in row:
        row[age_col] = 1
    tod_col = f"tod_{profile['time_of_day']}"
    if tod_col in row:
        row[tod_col] = 1
    dow_col = f"dow_{profile['day_of_week']}"
    if dow_col in row:
        row[dow_col] = 1
    row["is_weekend_contact"] = 1 if profile["day_of_week"] in ("sat", "sun") else 0

    for ch in CHANNELS:
        key = f"segment_rate_{ch}"
        if key in row:
            row[key] = segment_rates[profile["age_bracket"]][ch]

    return pd.DataFrame([row])[feature_cols]


def main():
    st.set_page_config(page_title="Contact Channel Recommender", layout="centered")
    st.title("Debtor Contact-Channel Recommender")
    st.caption(
        "Portfolio demo — trained entirely on synthetic data modeling publicly "
        "described debt-collection contact-optimization patterns. No real debtor "
        "data is used."
    )

    models, feature_cols, segment_rates = load_models()

    with st.form("profile_form"):
        col1, col2 = st.columns(2)
        with col1:
            age_bracket = st.selectbox("Age bracket", AGE_BRACKETS, index=1)
            debt_amount = st.number_input("Debt amount ($)", min_value=20, max_value=15000, value=850)
            days_since_last_contact = st.slider("Days since last contact", 0, 90, 10)
        with col2:
            prior_contact_attempts = st.slider("Prior contact attempts", 0, 15, 2)
            time_of_day = st.selectbox("Time of last contact", TIME_BUCKETS, index=1)
            day_of_week = st.selectbox("Day of last contact", DAYS_OF_WEEK, index=2)
        partial_payment_history = st.checkbox("Has made a partial payment before")

        submitted = st.form_submit_button("Get recommendation")

    if submitted:
        profile = {
            "age_bracket": age_bracket,
            "debt_amount": debt_amount,
            "days_since_last_contact": days_since_last_contact,
            "prior_contact_attempts": prior_contact_attempts,
            "time_of_day": time_of_day,
            "day_of_week": day_of_week,
            "partial_payment_history": int(partial_payment_history),
        }
        X = build_feature_row(profile, feature_cols, segment_rates)

        probs = {ch: models[ch].predict_proba(X.values)[:, 1][0] for ch in CHANNELS}
        best_channel = max(probs, key=probs.get)
        best_conf = probs[best_channel]
        needs_review = best_conf < CONFIDENCE_THRESHOLD

        st.subheader("Recommendation")
        if needs_review:
            st.warning(f"⚠️ Low confidence ({best_conf:.0%}) — flag for human review")
        else:
            st.success(f"✅ Contact via **{best_channel.upper()}** — {best_conf:.0%} confidence")

        st.write("Per-channel calibrated probability:")
        prob_df = pd.DataFrame({
            "channel": list(probs.keys()),
            "probability": [round(v, 3) for v in probs.values()],
        }).sort_values("probability", ascending=False)
        st.bar_chart(prob_df.set_index("channel"))
        st.dataframe(prob_df, hide_index=True)


if __name__ == "__main__":
    main()
