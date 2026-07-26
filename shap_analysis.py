"""
SHAP analysis for the loan denial model.

Two questions:
1. Which features drive denial predictions overall?
2. Since the model never sees race/sex/age, which features act as PROXIES
   and mechanically produce the group gap?
"""

import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

bundle = joblib.load("model.joblib")
model = bundle["model"]
NUMERIC = bundle["numeric"]
CATEGORICAL = bundle["categorical"]
FEATURES = NUMERIC + CATEGORICAL

# Rebuild the exact same test split used in training
df = pd.read_parquet("data/clean.parquet")
X = df[FEATURES].copy()
for c in CATEGORICAL:
    X[c] = X[c].astype("category")
y = df["denied"]
groups = df[["derived_race", "derived_sex", "applicant_age"]]

X_tr, X_te, y_tr, y_te, g_tr, g_te = train_test_split(
    X, y, groups, test_size=0.25, random_state=42, stratify=y
)

# ---------------------------------------------------------- SHAP values
print("Computing SHAP values on {:,} rows...".format(len(X_te)))
dm = xgb.DMatrix(X_te, enable_categorical=True)
contribs = model.get_booster().predict(dm, pred_contribs=True)
shap_vals = contribs[:, :-1]          # last column is the bias term
shap_df = pd.DataFrame(shap_vals, columns=FEATURES, index=X_te.index)

# ---------------------------------------------------------- 1. global
print("\n=== GLOBAL FEATURE IMPORTANCE (mean |SHAP|, log-odds) ===")
imp = shap_df.abs().mean().sort_values(ascending=False)
print(imp.round(4).to_string())

plt.figure(figsize=(8, 5))
imp.sort_values().plot(kind="barh", color="#4C72B0")
plt.xlabel("mean |SHAP| (log-odds)")
plt.title("What drives denial predictions")
plt.tight_layout()
plt.savefig("fig_global_importance.png", dpi=150)
plt.close()

# ---------------------------------------------------------- 2. gap decomposition
def decompose(attr, group_a, group_b):
    """How much of the log-odds gap between two groups does each feature explain?"""
    a = g_te[attr] == group_a
    b = g_te[attr] == group_b
    if a.sum() < 100 or b.sum() < 100:
        return None
    gap = shap_df[a].mean() - shap_df[b].mean()
    out = pd.DataFrame({
        "mean_shap_" + group_a[:12]: shap_df[a].mean().round(4),
        "mean_shap_" + group_b[:12]: shap_df[b].mean().round(4),
        "gap_contribution": gap.round(4),
        "pct_of_gap": (100 * gap / gap.abs().sum()).round(1),
    }).sort_values("gap_contribution", key=abs, ascending=False)
    print("\n=== GAP DECOMPOSITION: {} vs {} ({}) ===".format(group_a, group_b, attr))
    print("Total log-odds gap: {:+.4f}".format(gap.sum()))
    print(out.to_string())
    return out

decompose("derived_race", "Black or African American", "White")
decompose("derived_sex", "Female", "Joint")
decompose("applicant_age", ">74", "35-44")

# ---------------------------------------------------------- 3. feature values by group
print("\n=== UNDERLYING FEATURE MEDIANS BY RACE ===")
med = X_te[NUMERIC].join(g_te["derived_race"]).groupby("derived_race").median()
counts = g_te["derived_race"].value_counts()
med = med.loc[counts[counts >= 500].index]
print(med.round(3).to_string())

print("\nSaved fig_global_importance.png")
