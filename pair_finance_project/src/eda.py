"""
Step 3: Exploratory data analysis.
Checks class balance per channel and correlations between features/outcomes.
Saves plots to /home/claude/pair_finance_project/plots/.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

DATA_PATH = "/home/claude/pair_finance_project/data/synthetic_debtors.csv"
PLOTS_DIR = "/home/claude/pair_finance_project/plots"


def main():
    df = pd.read_csv(DATA_PATH)

    # --- Class balance per channel ---
    fig, ax = plt.subplots(figsize=(6, 4))
    rates = df[["responded_email", "responded_sms", "responded_post"]].mean()
    rates.index = ["Email", "SMS", "Post"]
    ax.bar(rates.index, rates.values, color=["#4C72B0", "#55A868", "#C44E52"])
    ax.set_ylabel("Response rate")
    ax.set_title("Class balance: response rate by channel")
    ax.set_ylim(0, 1)
    for i, v in enumerate(rates.values):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/class_balance.png", dpi=150)
    plt.close()

    # --- Response rate by age bracket and channel ---
    age_order = ["18-25", "26-35", "36-50", "51-65", "65+"]
    grouped = df.groupby("age_bracket")[
        ["responded_email", "responded_sms", "responded_post"]
    ].mean().reindex(age_order)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(age_order))
    width = 0.25
    ax.bar(x - width, grouped["responded_email"], width, label="Email", color="#4C72B0")
    ax.bar(x, grouped["responded_sms"], width, label="SMS", color="#55A868")
    ax.bar(x + width, grouped["responded_post"], width, label="Post", color="#C44E52")
    ax.set_xticks(x)
    ax.set_xticklabels(age_order)
    ax.set_ylabel("Response rate")
    ax.set_title("Response rate by age bracket and channel")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/response_by_age_channel.png", dpi=150)
    plt.close()

    # --- Correlation heatmap of numeric features vs outcomes ---
    numeric_cols = [
        "debt_amount", "days_since_last_contact", "prior_contact_attempts",
        "partial_payment_history", "responded_email", "responded_sms", "responded_post"
    ]
    corr = df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(numeric_cols)))
    ax.set_yticks(range(len(numeric_cols)))
    ax.set_xticklabels(numeric_cols, rotation=45, ha="right")
    ax.set_yticklabels(numeric_cols)
    for i in range(len(numeric_cols)):
        for j in range(len(numeric_cols)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("Feature/outcome correlation matrix")
    fig.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/correlation_matrix.png", dpi=150)
    plt.close()

    print("EDA complete. Plots saved to:", PLOTS_DIR)
    print("\nResponse rate by age bracket:\n", grouped)
    print("\nCorrelation matrix:\n", corr.round(2))


if __name__ == "__main__":
    main()
