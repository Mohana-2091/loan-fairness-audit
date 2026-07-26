# Loan Fairness Audit

A mortgage denial-risk model, an audit of how it behaves across demographic groups, and a scored API that returns reason codes for every decision.

Built on **2023 HMDA data** (Home Mortgage Disclosure Act) — real mortgage applications that US lenders are federally required to report, including applicant race, sex and age. 219,305 owner-occupied single-family applications from Ohio after cleaning.

The point of the project is not the model. It is what the audit finds after the model is built.

---

## The question

A lender wants to automate first-pass mortgage screening. The obvious safeguard is to exclude race, sex and age from the model — and that is what this model does.

Does excluding them actually produce fair outcomes?

---

## What the data looks like before any modelling

Raw denial rates in the Ohio 2023 population:

| Group | Denial rate | n |
|---|---|---|
| White | 20.5% | 189,162 |
| Black or African American | 35.3% | 18,739 |
| Asian | 20.8% | 6,446 |
| American Indian / Alaska Native | 41.2% | 762 |

| Group | Denial rate |
|---|---|
| Joint applicants | 16.5% |
| Male | 24.4% |
| Female | 25.9% |

These gaps are not evidence of discrimination on their own — applicants differ on income, loan-to-value and debt-to-income. The audit exists to separate the two.

---

## The model

XGBoost binary classifier. Features: loan amount, income, property value, loan-to-value ratio, loan-to-income ratio, loan term, loan type, loan purpose, lien status, debt-to-income band.

**Race, sex, ethnicity and age are excluded from training.**

| Metric | Value |
|---|---|
| ROC-AUC | 0.879 |
| PR-AUC | 0.770 |
| Base denial rate | 21.8% |

The decision threshold is set so the model refers the same volume the market actually denied, rather than at an arbitrary 0.5.

---

## Finding 1 — Excluding protected attributes does not remove disparity

The model never sees race, yet reproduces the disparity almost exactly:

| Group | Predicted denial rate | Actual denial rate | FPR |
|---|---|---|---|
| White | 20.8% | 20.4% | 0.092 |
| Black or African American | 33.9% | 35.6% | 0.137 |
| Asian | 17.5% | 20.3% | 0.046 |
| Joint | 16.3% | 18.5% | 0.065 |

| Constraint | Race | Sex | Age |
|---|---|---|---|
| Demographic parity ratio | 0.482 | 0.608 | 0.264 |
| Four-fifths rule (≥ 0.80) | fail | fail | fail |

**The false positive rate is the number that matters.** FPR here means: of applicants who were actually approved, what share does the model flag for denial? Creditworthy applicants wrongly referred.

- Black applicants: **1.5×** the White rate, **3.0×** the Asian rate
- Applicants over 74: **6.6×** the rate of applicants under 25

This is *disparate impact* — bias transmitted through features correlated with race rather than through race itself.

## Finding 2 — Which features carry it

Median values in the test set:

| | White | Black or African American |
|---|---|---|
| Income | $84k | $69k |
| Property value | $255k | $205k |
| Loan-to-value ratio | 80.0 | 83.7 |
| Loan-to-income ratio | 1.65 | 1.82 |

Income, loan-to-value and loan-to-income are legitimate credit factors. They are also correlated with race, because the wealth gap is already encoded in them. Removing race from the feature list does not remove race from the data.

## Finding 3 — Mitigation is not free, and the accuracy metric hides the cost

Applying an equalized-odds constraint (fairlearn `ThresholdOptimizer`):

| Metric | Baseline | Mitigated | Change |
|---|---|---|---|
| FPR gap (max/min) | 2.99× | 1.43× | **−52%** |
| Equalized odds difference | 0.109 | 0.033 | **−70%** |
| Demographic parity ratio | 0.482 | 0.603 | +25% |
| Accuracy | 0.854 | 0.872 | +1.7 pts |

Accuracy went **up** while fairness improved. That should be treated as a warning, not a result.

Recall tells the real story:

| Group | Baseline TPR | Mitigated TPR |
|---|---|---|
| White | 0.659 | 0.493 |
| Black or African American | 0.705 | 0.484 |
| Asian | 0.684 | 0.517 |

Recall drops roughly 17 points. The constrained model refers far fewer applications overall, so it catches fewer genuine denials — the lender absorbs more risk. With a 21.8% base rate, accuracy rewards predicting the majority class, which is exactly what happened. **The cost is in recall, not accuracy.** 9.36% of individual decisions change.

## Finding 4 — The mitigation cannot legally be deployed as-is

`ThresholdOptimizer` requires the protected attribute *at inference time* to apply group-specific thresholds. Setting different approval thresholds by race is disparate treatment, which is prohibited under US fair lending law.

So the technique is a useful **audit and diagnostic** tool, not a shippable fix. The fair-lending bind is structural: exclude race and you get disparate impact; use race to correct it and you get disparate treatment. Real remediation lives in feature selection, data collection and human review design — not in post-processing.

---

## The API

`POST /predict` returns a denial probability, the decision at the audited threshold, and the top five SHAP contributions in log-odds — the reason-code structure that adverse action notices require.

```json
{
  "denial_probability": 0.1433,
  "threshold": 0.2973,
  "decision": "pass",
  "top_reasons": [
    {"feature": "loan_purpose", "value": "1", "shap_log_odds": -1.2372, "direction": "reduces risk"},
    {"feature": "debt_to_income_ratio", "value": "36", "shap_log_odds": -0.5632, "direction": "reduces risk"},
    {"feature": "loan_to_value_ratio", "value": "83.7", "shap_log_odds": 0.1, "direction": "increases risk"}
  ]
}
```

`GET /health` returns liveness and the active threshold.

The model ships in XGBoost's native JSON format rather than a pickle, so the serving container is not coupled to the library version used in training.

---

## Running it

```bash
# 1. environment
python -m venv venv && source venv/bin/activate     # Windows: venv\Scripts\activate
pip install -r requirements-dev.txt

# 2. data
# download 2023 HMDA state data from https://ffiec.cfpb.gov/data-browser/
# place the CSV at data/state_OH.csv

# 3. pipeline
python prep_data.py         # clean, scope, derive features
python train_model.py       # train + fairness audit
python shap_analysis.py     # feature attribution and gap decomposition
python mitigate.py          # constrained model + trade-off table
python export_artifacts.py  # model.json + meta.json for serving

# 4. serve
uvicorn main:app --reload --port 8000
# or
docker build -t loan-fairness-api .
docker run -p 8080:8000 loan-fairness-api
```

Docs at `/docs`.

---

## Stack

Python · pandas · scikit-learn · XGBoost · SHAP · fairlearn · FastAPI · Docker · GitHub Actions

## Repository

| File | Purpose |
|---|---|
| `prep_data.py` | Cleaning, scoping, feature derivation |
| `train_model.py` | Training and the fairness audit |
| `shap_analysis.py` | Global attribution and group-gap decomposition |
| `mitigate.py` | Equalized-odds constraint and trade-off table |
| `export_artifacts.py` | Portable model export |
| `main.py` | FastAPI service |
| `Dockerfile` | Container definition |

---

## Scope and limitations

- One state, one year. Findings are specific to Ohio 2023 and do not generalise without re-running.
- HMDA records the outcome, not the underwriting file. Credit score, reserves and employment history are absent, so some of the residual gap reflects omitted variables rather than bias.
- Denial is a proxy for creditworthiness that already embeds lender behaviour. A model trained on it learns the market's decisions, not ground truth.
- This is a research and audit demonstration on public data. It is not a credit decisioning system.
