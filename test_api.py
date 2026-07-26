"""API tests. These run in CI against the exported artifacts."""

import pytest
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)

VALID = {
    "loan_amount": 105000,
    "income": 69,
    "property_value": 205000,
    "loan_to_value_ratio": 83.7,
    "loan_term": 360,
    "loan_type": 1,
    "loan_purpose": 1,
    "lien_status": 1,
    "debt_to_income_ratio": "36",
}


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_returns_probability_and_reasons():
    r = client.post("/predict", json=VALID)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["denial_probability"] <= 1.0
    assert body["decision"] in {"pass", "refer for review"}
    assert len(body["top_reasons"]) == 5
    for reason in body["top_reasons"]:
        assert reason["direction"] in {"increases risk", "reduces risk"}


def test_decision_matches_threshold():
    body = client.post("/predict", json=VALID).json()
    expected = "refer for review" if body["denial_probability"] >= body["threshold"] else "pass"
    assert body["decision"] == expected


def test_higher_ltv_does_not_reduce_risk():
    """Sanity check on model direction: a much higher LTV should not score safer."""
    low = client.post("/predict", json={**VALID, "loan_to_value_ratio": 60.0}).json()
    high = client.post("/predict", json={**VALID, "loan_to_value_ratio": 97.0}).json()
    assert high["denial_probability"] >= low["denial_probability"] - 0.05


@pytest.mark.parametrize("field,bad", [
    ("loan_amount", -1),
    ("income", 0),
    ("loan_to_value_ratio", 500),
])
def test_rejects_invalid_input(field, bad):
    r = client.post("/predict", json={**VALID, field: bad})
    assert r.status_code == 422


def test_unseen_category_does_not_crash():
    r = client.post("/predict", json={**VALID, "debt_to_income_ratio": "not-a-real-band"})
    assert r.status_code == 200
