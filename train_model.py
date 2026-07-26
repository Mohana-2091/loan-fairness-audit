"""
Train a denial-prediction model WITHOUT protected attributes,
then audit its behaviour across protected groups.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from xgboost import XGBClassifier
from fairlearn.metrics import (
    MetricFrame,
    selection_rate,
    false_positive_rate,
    true_positive_rate,
    demographic_parity_ratio,
    equalized_odds_difference,
)

# ---------------------------------------------------------------- load
df = pd.read_parquet("data/clean.parquet")
print("Rows:", len(df))

PROTECTED = ["derived_race", "derived_sex", "applicant_age", "derived_ethnicity"]

NUMERIC = ["loan_amount", "loan_to_value_ratio", "income",
           "property_value", "loan_term", "loan_to_income"]
CATEGORICAL = ["loan_type", "loan_purpose", "lien_status", "debt_to_income_ratio"]

X = df[NUMERIC + CATEGORICAL].copy()
for c in CATEGORICAL:
    X[c] = X[c].astype("category")

y = df["denied"]
groups = df[PROTECTED]

# ---------------------------------------------------------------- split
X_tr, X_te, y_tr, y_te, g_tr, g_te = train_test_split(
    X, y, groups, test_size=0.25, random_state=42, stratify=y
)

# ---------------------------------------------------------------- train
model = XGBClassifier(
    n_estimators=400,
    max_depth=5,
    learning_rate=0.08,
    subsample=0.8,
    colsample_bytree=0.8,
    enable_categorical=True,
    tree_method="hist",
    eval_metric="auc",
    random_state=42,
)
model.fit(X_tr, y_tr)

proba = model.predict_proba(X_te)[:, 1]

print("\n=== MODEL PERFORMANCE ===")
print("ROC-AUC : {:.4f}".format(roc_auc_score(y_te, proba)))
print("PR-AUC  : {:.4f}".format(average_precision_score(y_te, proba)))
print("Base denial rate: {:.1%}".format(y_te.mean()))

# Threshold chosen so the model flags the same volume the market actually denied
THRESHOLD = float(np.quantile(proba, 1 - y_te.mean()))
pred = (proba >= THRESHOLD).astype(int)
print("Threshold: {:.4f}".format(THRESHOLD))

# ---------------------------------------------------------------- audit
metrics = {
    "predicted_denial_rate": selection_rate,
    "actual_denial_rate": lambda yt, yp: yt.mean(),
    "recall_TPR": true_positive_rate,
    "FPR": false_positive_rate,
}

for attr in ["derived_race", "derived_sex", "applicant_age"]:
    sf = g_te[attr]
    mf = MetricFrame(metrics=metrics, y_true=y_te, y_pred=pred, sensitive_features=sf)
    out = mf.by_group.copy()
    out["n"] = sf.value_counts().reindex(out.index)
    out = out[out["n"] >= 200]

    print("\n=== FAIRNESS BY {} ===".format(attr.upper()))
    print(out.sort_values("n", ascending=False).round(4).to_string())

    keep = sf.isin(out.index)
    print("Demographic parity ratio : {:.3f}".format(
        demographic_parity_ratio(y_te[keep], pred[keep], sensitive_features=sf[keep])))
    print("Equalized odds difference: {:.3f}".format(
        equalized_odds_difference(y_te[keep], pred[keep], sensitive_features=sf[keep])))

# ---------------------------------------------------------------- save
joblib.dump({"model": model, "threshold": THRESHOLD,
             "numeric": NUMERIC, "categorical": CATEGORICAL},
            "model.joblib")

X_te.assign(denied=y_te.values, proba=proba, pred=pred).join(g_te).to_parquet(
    "data/audit_test.parquet", index=False)

print("\nSaved model.joblib and data/audit_test.parquet")
