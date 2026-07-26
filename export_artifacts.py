"""
Export deployment artifacts.

Pickled sklearn/XGBoost objects are tied to the library version that created
them, so a model pickled locally will not load inside a container built on a
different XGBoost release. Instead we export:

  model.json  - XGBoost native format, portable across versions
  meta.json   - threshold, feature order, and training category levels

This is the version-portable way to ship a tree model.
"""

import json
import joblib
import numpy as np
import pandas as pd

bundle = joblib.load("model.joblib")
model = bundle["model"]
NUMERIC = bundle["numeric"]
CATEGORICAL = bundle["categorical"]

# ------------------------------------------------- model in native format
model.get_booster().save_model("model.json")
print("Saved model.json (XGBoost native format)")

# ------------------------------------------------- category levels
df = pd.read_parquet("data/clean.parquet", columns=CATEGORICAL)


def to_native(v):
    if isinstance(v, np.generic):
        return v.item()
    return v


categories = {}
for c in CATEGORICAL:
    cats = [to_native(v) for v in pd.Categorical(df[c]).categories]
    categories[c] = cats
    print("{:26s} {:3d} levels  e.g. {}".format(c, len(cats), cats[:4]))

meta = {
    "threshold": float(bundle["threshold"]),
    "numeric": list(NUMERIC),
    "categorical": list(CATEGORICAL),
    "categories": categories,
}

with open("meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print("\nSaved meta.json")
