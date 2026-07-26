import pandas as pd
import numpy as np

COLS = [
    "action_taken", "derived_race", "derived_sex", "derived_ethnicity",
    "loan_amount", "loan_to_value_ratio", "income", "debt_to_income_ratio",
    "property_value", "loan_term", "loan_type", "loan_purpose",
    "lien_status", "occupancy_type", "total_units",
    "business_or_commercial_purpose", "applicant_age",
]

print("Loading...")
df = pd.read_csv("data/state_OH.csv", usecols=COLS, low_memory=False)
print("Raw:", df.shape)

# Target: 1 = originated (approved), 3 = denied
df = df[df["action_taken"].isin([1, 3])].copy()
df["denied"] = (df["action_taken"] == 3).astype(int)

# Scope: standard owner-occupied single-family consumer mortgages
df = df[df["business_or_commercial_purpose"] == 2]
df = df[df["occupancy_type"] == 1]
df = df[df["total_units"].astype(str) == "1"]

# Numeric cleanup
for c in ["loan_amount", "income", "property_value", "loan_to_value_ratio", "loan_term"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df = df.dropna(subset=["loan_amount", "income", "property_value"])
df = df[(df["income"] > 0) & (df["income"] < 2000)]  # income reported in $000s
df = df[df["loan_to_value_ratio"].between(1, 200)]

# Protected attributes - keep only usable labels
df = df[~df["derived_race"].isin(["Race Not Available", "Free Form Text Only"])]
df = df[~df["derived_sex"].isin(["Sex Not Available"])]
df = df[df["applicant_age"].astype(str) != "8888"]

# Derived feature
df["loan_to_income"] = df["loan_amount"] / (df["income"] * 1000)

df = df.drop(columns=["action_taken", "business_or_commercial_purpose",
                      "occupancy_type", "total_units"])

df.to_parquet("data/clean.parquet", index=False)

print("\nClean:", df.shape)
print("\nOverall denial rate: {:.1%}".format(df["denied"].mean()))

print("\nDenial rate by race:")
print(df.groupby("derived_race")["denied"].agg(["mean", "count"]).sort_values("count", ascending=False))

print("\nDenial rate by sex:")
print(df.groupby("derived_sex")["denied"].agg(["mean", "count"]))

print("\nDenial rate by age group:")
print(df.groupby("applicant_age")["denied"].agg(["mean", "count"]))
