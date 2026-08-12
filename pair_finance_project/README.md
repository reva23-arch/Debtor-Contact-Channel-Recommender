# Debtor Contact-Channel Recommender (synthetic data portfolio project)

Given a debtor's behavioral and contact history, predict the probability they
respond within N days for each of 3 contact strategies (email, SMS, post),
and flag predictions where confidence is too low to act on automatically.

This models the kind of per-channel response-likelihood decision that
contact-optimization systems in debt collection (e.g. PAIR Finance, publicly
described) are built around — **not** "will they repay," but "which channel,
and how sure are we." **All data here is synthetic**, generated to encode
domain-plausible patterns; no real debtor data is used or represented
anywhere in this project.

## Why this project, not just another classifier

Most portfolio ML projects stop at "train a model, report AUROC." The part
that actually mirrors production decision-making in a regulated space like
debt collection is **knowing when not to trust the model** — hence the
calibration + decision layer, which is the differentiating piece here.

## Pipeline

| Step | File | What it does |
|---|---|---|
| 1 | (this README) | Problem framing |
| 2 | `src/generate_data.py` | Synthetic 4,000-debtor dataset with realistic embedded patterns + noise |
| 3 | `src/eda.py` | Class balance, segment breakdowns, correlation matrix (see `plots/`) |
| 4 | `src/features.py` | One-hot encoding, engineered features, **time-based** train/test split |
| 5 | `src/train_model.py` | One `HistGradientBoostingClassifier` per channel |
| 6 | `src/train_model.py` | Isotonic calibration on a held-out calibration slice + reliability diagrams |
| 7 | `src/train_model.py` | Decision layer: recommended channel + confidence + human-review flag |
| 8 | `app.py` | Streamlit demo: input a profile → get a recommendation |

## Dataset

4,000 synthetic debtor profiles with: age bracket, debt amount, days since
last contact, prior contact attempts, time-of-day/day-of-week of last
contact, and partial-payment history. Response outcomes per channel are
generated from a logistic model with domain-plausible coefficients (younger
brackets respond better to SMS, older brackets to post, higher debt amounts
and higher contact-attempt counts correlate with slower response, evening
contact has better open rates than night) **plus substantial random noise**,
so the signal is real but the problem isn't trivially separable — same as
real-world response data.

Run it: `python3 src/generate_data.py`

## Modeling notes (the part worth reading if you're evaluating this)

- **Time-based split, not random.** Debtors are split by simulated recency
  (test = most-recently-contacted cohort) rather than a random 80/20, because
  a system like this is deployed forward in time — a random split would
  overstate real-world performance.
- **Calibration is done correctly, not just "run `CalibratedClassifierCV`."**
  The first version of this pipeline used `CalibratedClassifierCV(cv=3)`,
  which internally refits the base model on small folds and *destroyed* the
  ranking (AUROC collapsed to ~0.50 across all channels — no better than
  random). The fix: fit the base model on a training split, hold out a
  separate calibration slice it never saw, and calibrate on that using
  `FrozenEstimator`. This recovered the model's real AUROC (0.53–0.57,
  consistent with the injected noise level) *and* improved Brier scores
  across all three channels. This kind of failure mode — a calibration step
  that silently destroys discrimination if wired up naively — is exactly
  the kind of thing worth catching before it ships.
- **Modest AUROC is intentional, not a bug.** 0.53–0.57 reflects the noise
  level deliberately built into the label-generating process (real
  contact-response data is noisy; a model that reports 0.95 AUROC on this
  kind of data would be a red flag, not a win). What matters more here is
  that the *calibration curve* tracks the diagonal reasonably well — when
  the model says "70% confident," it should be right about 70% of the time.

| Channel | AUROC | Brier (calibrated) |
|---|---|---|
| Email | 0.568 | 0.223 |
| SMS   | 0.544 | 0.263 |
| Post  | 0.535 | 0.258 |

Reliability diagrams (raw vs. calibrated) are in `plots/reliability_diagrams.png`.

## Decision layer

For each debtor, take the highest calibrated probability across the three
channels. If it clears the confidence threshold (0.60, chosen from the
observed confidence distribution — see below), auto-recommend that channel.
Otherwise, flag for human review. On the held-out test set, **37.5%** of
cases fall below threshold and get flagged — a defensible number for a
first-pass system, not a rubber stamp.

In production, this threshold is a business decision, not just a stats one:
it trades off auto-action coverage against the cost of a wrong auto-contact
(compliance risk, wasted contact attempts) vs. the cost of a human review
queue. This repo picks a threshold from the data distribution as a
reasonable starting point; a real deployment would tune it against a defined
cost function and an ops team's review-queue capacity.

## Running it

```bash
pip install -r requirements.txt   # or: pandas numpy scikit-learn matplotlib joblib streamlit

python3 src/generate_data.py      # Step 2: generate synthetic data
python3 src/eda.py                # Step 3: EDA + plots
python3 src/features.py           # Step 4: feature engineering + split
python3 src/train_model.py        # Steps 5-7: train, calibrate, evaluate, decide

streamlit run app.py              # Step 8: interactive demo
```

## What would need to change for real production data

- Feature set would expand (actual contact-response logs, not simulated
  ones; possibly text/sentiment from prior interactions if compliant to use)
- The synthetic label-generating patterns (age→channel affinity, etc.) would
  need to be validated against real response data — they're informed guesses
  based on publicly available domain patterns, not fitted to anything real
- Fairness/compliance review would be required before using age as a
  feature in a regulated debt-collection context — flagged here explicitly
  since this is exactly the kind of decision a real deployment can't skip
- The confidence threshold would be tuned against an actual cost function
  and ops review-queue capacity, not just the observed probability
  distribution

## Project structure

```
pair_finance_project/
├── README.md
├── app.py                      # Streamlit demo
├── src/
│   ├── generate_data.py        # Step 2
│   ├── eda.py                  # Step 3
│   ├── features.py             # Step 4
│   └── train_model.py          # Steps 5-7
├── data/                       # generated CSVs + metrics (gitignored in real repo)
├── models/                     # saved calibrated models + feature schema
└── plots/                      # EDA + reliability diagram PNGs
```

## 60-second pitch (for interviews)

> "I built a synthetic model of a debtor population and a calibrated
> classifier predicting per-channel contact response likelihood, with a
> confidence threshold for flagging low-certainty cases for human review —
> same evaluation approach I used in my anomaly detection project, just
> applied to a different domain. The interesting bug I hit and fixed:
> naively wiring up scikit-learn's calibration wrapper silently destroyed
> the model's discrimination — AUROC dropped to random — because it was
> refitting on small internal folds. Fixing it meant calibrating on a
> proper held-out slice instead."
