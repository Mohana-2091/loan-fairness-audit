"""
Loan Denial Risk API with reason codes.

POST /predict   ->  denial probability, decision at the audited threshold,
                    and the top contributing factors (SHAP, log-odds).

Design notes
------------
* The model is trained WITHOUT race, sex, ethnicity or age. The audit that
  accompanies this service (see README) shows it still reproduces group-level
  disparity through correlated features, so every response carries reason
  codes to make the decision reviewable.
* The model is loaded from XGBoost's native JSON format rather than a pickle,
  so the container is not tied to the library version used in training.
"""

import json
import traceback
from typing import List, Literal, Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

with open("meta.json") as f:
    META = json.load(f)

THRESHOLD = float(META["threshold"])
NUMERIC = META["numeric"]
CATEGORICAL = META["categorical"]
CATEGORIES = META["categories"]
FEATURES = NUMERIC + CATEGORICAL

BOOSTER = xgb.Booster()
BOOSTER.load_model("model.json")

app = FastAPI(
    title="Loan Denial Risk API",
    description="Denial risk scoring with SHAP reason codes and a documented fairness audit.",
    version="1.0.0",
)


class Application(BaseModel):
    loan_amount: float = Field(..., gt=0, examples=[105000])
    income: float = Field(..., gt=0, description="Annual income in thousands", examples=[69])
    property_value: float = Field(..., gt=0, examples=[205000])
    loan_to_value_ratio: float = Field(..., gt=0, le=200, examples=[83.7])
    loan_term: int = Field(360, gt=0, examples=[360])
    loan_type: int = Field(1, description="1 Conventional, 2 FHA, 3 VA, 4 RHS/FSA", examples=[1])
    loan_purpose: int = Field(1, description="1 Purchase, 2 Improvement, 31 Refi, 32 Cash-out", examples=[1])
    lien_status: int = Field(1, description="1 First lien, 2 Subordinate", examples=[1])
    debt_to_income_ratio: str = Field("36", examples=["36"])


class Reason(BaseModel):
    feature: str
    value: Optional[str]
    shap_log_odds: float
    direction: Literal["increases risk", "reduces risk"]


class Prediction(BaseModel):
    denial_probability: float
    threshold: float
    decision: Literal["refer for review", "pass"]
    top_reasons: List[Reason]
    disclaimer: str


def build_frame(app_in: Application) -> pd.DataFrame:
    d = app_in.model_dump()
    d["loan_to_income"] = d["loan_amount"] / (d["income"] * 1000)

    frame = pd.DataFrame({c: [float(d[c])] for c in NUMERIC})

    for c in CATEGORICAL:
        cats = list(CATEGORIES[c])
        val = d[c]
        if cats and isinstance(cats[0], str):
            val = str(val)
        else:
            try:
                val = type(cats[0])(val)
            except (ValueError, TypeError):
                val = None
        if val not in cats:
            val = None  # unseen level -> treated as missing by XGBoost
        frame[c] = pd.Categorical([val], categories=cats)

    return frame[FEATURES]


def to_text(v) -> Optional[str]:
    """Numpy / categorical scalars are not JSON-serialisable. Render as text."""
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, float):
        return "{:g}".format(v)
    return str(v)


@app.get("/health")
def health():
    return {"status": "ok", "threshold": round(THRESHOLD, 4), "n_features": len(FEATURES)}


@app.post("/predict", response_model=Prediction)
def predict(application: Application):
    try:
        frame = build_frame(application)
        dm = xgb.DMatrix(frame, enable_categorical=True)

        proba = float(np.asarray(BOOSTER.predict(dm))[0])
        contribs = np.asarray(BOOSTER.predict(dm, pred_contribs=True))[0][:-1]
        order = np.argsort(np.abs(contribs))[::-1][:5]

        reasons = [
            Reason(
                feature=FEATURES[i],
                value=to_text(frame.iloc[0][FEATURES[i]]),
                shap_log_odds=round(float(contribs[i]), 4),
                direction="increases risk" if contribs[i] > 0 else "reduces risk",
            )
            for i in order
        ]

        return Prediction(
            denial_probability=round(proba, 4),
            threshold=round(THRESHOLD, 4),
            decision="refer for review" if proba >= THRESHOLD else "pass",
            top_reasons=reasons,
            disclaimer=(
                "Trained on public HMDA data for research and audit demonstration. "
                "Not a credit decision. See README for the documented fairness audit."
            ),
        )

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="{}: {}".format(type(e).__name__, e))
