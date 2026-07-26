"""
Bias mitigation: apply a fairness constraint and measure what it costs.

Baseline  = single global threshold, no fairness constraint
Mitigated = fairlearn ThresholdOptimizer under an equalized-odds constraint

The output is a trade-off table: how much accuracy is given up to close
the false-positive-rate gap between groups.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.metrics import (
    MetricFrame, selection_rate, false_positive_rate, true_positive_rate,
    demographic_parity_ratio, equalized_odds_difference,
)

bundle = joblib.load("model.joblib")
model = bundle["model"]
THRESHOLD = bundle["threshold"]
FEATURES = bundle["numeric"] + bundle["categorical"]
CATEGORICAL = bundle["categorical"]


class Float64Estimator(ClassifierMixin, BaseEstimator):
    """XGBoost returns float32 probabilities, which recent pandas versions
    refuse to upcast inside fairlearn. Cast to float64 at the boundary.

    Inherits from BaseEstimator so sklearn 1.6+ tag machinery works.
    """

    def __init__(self, model=None):
        self.model = model

    def fit(self, X, y=None, **kwargs):
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        return np.asarray(self.model.predict_proba(X), dtype=np.float64)

    def predict(self, X):
        return np.asarray(self.model.predict(X), dtype=np.float64)

    def __sklearn_is_fitted__(self):
        return True


wrapped = Float64Estimator(model=model)
wrapped.classes_ = np.array([0, 1])

df = pd.read_parquet("data/clean.parquet")
X = df[FEATURES].copy()
for c in CATEGORICAL:
    X[c] = X[c].astype("category")
y = df["denied"].astype(int)
race = df["derived_race"].astype(str)

X_tr, X_te, y_tr, y_te, r_tr, r_te = train_test_split(
    X, y, race, test_size=0.25, random_state=42, stratify=y
)

# Restrict the audit to groups with enough test volume to measure reliably
big = r_te.value_counts()
big = big[big >= 500].index

keep_te = r_te.isin(big)
keep_tr = r_tr.isin(big)

Xte = X_te[keep_te].reset_index(drop=True)
yte = y_te[keep_te].reset_index(drop=True)
rte = r_te[keep_te].reset_index(drop=True)

Xtr = X_tr[keep_tr].reset_index(drop=True)
ytr = y_tr[keep_tr].reset_index(drop=True)
rtr = r_tr[keep_tr].reset_index(drop=True)

METRICS = {
    "denial_rate": selection_rate,
    "TPR": true_positive_rate,
    "FPR": false_positive_rate,
}


def report(name, y_true, y_pred, sf):
    y_pred = np.asarray(y_pred).astype(int)
    mf = MetricFrame(metrics=METRICS, y_true=y_true, y_pred=y_pred, sensitive_features=sf)
    tbl = mf.by_group.copy()
    tbl["n"] = sf.value_counts().reindex(tbl.index)
    print("\n=== {} ===".format(name))
    print(tbl.sort_values("n", ascending=False).round(4).to_string())
    dp = demographic_parity_ratio(y_true, y_pred, sensitive_features=sf)
    eo = equalized_odds_difference(y_true, y_pred, sensitive_features=sf)
    acc = accuracy_score(y_true, y_pred)
    fpr_gap = tbl["FPR"].max() / max(tbl["FPR"].min(), 1e-9)
    print("accuracy {:.4f} | demographic parity ratio {:.3f} | "
          "equalized odds diff {:.3f} | FPR max/min {:.2f}x".format(acc, dp, eo, fpr_gap))
    return {"accuracy": acc, "parity_ratio": dp, "eq_odds_diff": eo, "fpr_ratio": fpr_gap}


# ------------------------------------------------------------ baseline
proba = wrapped.predict_proba(Xte)[:, 1]
base_pred = (proba >= THRESHOLD).astype(int)
print("Model ranking quality (AUC, unchanged by thresholding): {:.4f}".format(
    roc_auc_score(yte, proba)))
base = report("BASELINE - single threshold, no constraint", yte, base_pred, rte)

# ------------------------------------------------------------ mitigated
print("\nFitting ThresholdOptimizer (equalized odds)...")
opt = ThresholdOptimizer(
    estimator=wrapped,
    constraints="equalized_odds",
    objective="accuracy_score",
    prefit=True,
    predict_method="predict_proba",
)
opt.fit(Xtr, ytr, sensitive_features=rtr)
mit_pred = opt.predict(Xte, sensitive_features=rte, random_state=42)
mit = report("MITIGATED - equalized odds constraint", yte, mit_pred, rte)

# ------------------------------------------------------------ trade-off
print("\n" + "=" * 62)
print("TRADE-OFF SUMMARY")
print("=" * 62)
rows = []
for k, label in [("accuracy", "Accuracy"), ("parity_ratio", "Demographic parity ratio"),
                 ("eq_odds_diff", "Equalized odds difference"), ("fpr_ratio", "FPR gap (max/min)")]:
    rows.append({"metric": label, "baseline": round(base[k], 4),
                 "mitigated": round(mit[k], 4), "change": round(mit[k] - base[k], 4)})
summary = pd.DataFrame(rows)
print(summary.to_string(index=False))
summary.to_csv("tradeoff_summary.csv", index=False)

flipped = (base_pred != np.asarray(mit_pred).astype(int)).mean()
print("\nShare of decisions that changed: {:.2%}".format(flipped))
print("Saved tradeoff_summary.csv")
