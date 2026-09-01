# Loan Fairness Audit — SAS Version

This is a SAS implementation of the same fair lending audit logic as the main
Python project in this repository: build a loan approval model that excludes
protected attributes, then audit its outputs for disparate impact under the
four-fifths rule, and evaluate a group-level threshold correction against the
fairness/accuracy trade-off it introduces.

## What it does

1. **Simulates a mortgage application dataset**, structured like HMDA data,
   with protected attributes (`protected_race`, `protected_sex`) generated
   but deliberately **excluded** from the model itself.
2. **Builds a loan approval model** (`PROC LOGISTIC`) using only legitimate
   underwriting features — income, loan amount, credit score,
   debt-to-income, loan-to-value, and neighborhood income tier.
3. **Audits the model's real-world outputs by protected group**, computing
   each group's approval rate and applying the **four-fifths rule** (a
   group's approval rate must be at least 80% of the highest-approval
   group's rate, or it signals disparate impact).
4. **Finds a real disparate-impact violation** even with protected
   attributes excluded — a correlated proxy feature (neighborhood income
   tier) reintroduces the bias, the same finding as the Python version.
5. **Applies an equalized-odds-style correction** — a group-specific
   decision threshold — to close the fairness gap, and reports the
   resulting approval rates and four-fifths ratios after correction.
6. **Reports the honest trade-off**: recall before vs. after the fairness
   correction, rather than treating the fix as a free win.

## How this maps to the Python version

| Concept                          | Python                                    | SAS                                      |
|-----------------------------------|--------------------------------------------|--------------------------------------------|
| Data handling                     | `pandas`                                   | `data` step                                |
| Model                              | `XGBoost` / `scikit-learn`                 | `PROC LOGISTIC`                            |
| Model evaluation (ROC-AUC)         | `sklearn.metrics.roc_auc_score`            | `PROC LOGISTIC` with `roc;` statement      |
| Group-level aggregation            | `pandas.groupby()`                         | `PROC SQL` with `GROUP BY`                 |
| Four-fifths rule check             | Custom fairness function                   | `PROC SQL` correlated subquery             |
| Equalized-odds correction          | `Fairlearn`'s constraint solver             | Group-specific threshold via `MERGE`       |
| Explainability                     | SHAP reason codes                          | Coefficient estimates / odds ratios        |

## A real bug hit and fixed along the way

The first version of this script had an **unsorted-merge bug** — SAS's
`MERGE ... BY` requires both datasets to already be sorted by the merge key.
Without `PROC SORT` first, the merge silently produced garbage (a 0%/100%
split instead of a real three-group comparison). Adding `PROC SORT` before
each `MERGE` step fixed it. This is one of the most common real-world SAS
mistakes, and debugging it was part of genuinely learning the language rather
than just running someone else's working code.

## Running it

Requires access to SAS (this was developed and tested on **SAS OnDemand for
Academics**, free at [welcome.oda.sas.com](https://welcome.oda.sas.com)).

Open `loan_fairness_audit.sas` in SAS Studio and run the full program (F3 or
the Run icon).

## Note on data

This version uses a simulated dataset with the same structure and predictive
signal as HMDA mortgage data (rather than the actual HMDA dataset), since the
goal was to demonstrate the SAS fairness-auditing workflow itself. The Python
version in the parent directory uses the actual HMDA dataset (219K
applications).
