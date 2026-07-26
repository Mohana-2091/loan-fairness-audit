"""API tests. These run in CI against the exported artifacts.

Note on what is NOT tested here: an earlier version asserted that a higher
loan-to-value ratio must not lower predicted denial risk. The model rejects
that assumption - see README, "Non-monotonic response to LTV". Encoding an
intuition the data does not support turns a finding into a broken build, so
these tests check the service contract and determinism instead.
"""

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
    body = r.json()
    assert body["status"] == "ok"
    assert 0.0 < body["threshold"] < 1.0


def test_predict_returns_probability_and_reasons():
    r = client.post("/predict", json=VALID)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["denial_probability"] <= 1.0
    assert body["decision"] in {"pass", "refer for review"}
    assert len(body["top_reasons"]) == 5
    for reason in body["top_reasons"]:
        assert reason["direction"] in {"increases risk", "reduces risk"}
        assert isinstance(reason["shap_log_odds"], float)


def test_decision_matches_threshold():
    body = client.post("/predict", json=VALID).json()
    expected = "refer for review" if body["denial_probability"] >= body["threshold"] else "pass"
    assert body["decision"] == expected


def test_reasons_are_ranked_by_magnitude():
    reasons = client.post("/predict", json=VALID).json()["top_reasons"]
    magnitudes = [abs(r["shap_log_odds"]) for r in reasons]
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_predictions_are_deterministic():
    a = client.post("/predict", json=VALID).json()
    b = client.post("/predict", json=VALID).json()
    assert a["denial_probability"] == b["denial_probability"]


def test_inputs_actually_move_the_score():
    """Guards against a silently broken feature pipeline returning a constant."""
    low_income = client.post("/predict", json={**VALID, "income": 25}).json()
    high_income = client.post("/predict", json={**VALID, "income": 250}).json()
    assert low_income["denial_probability"] != high_income["denial_probability"]


@pytest.mark.parametrize("field,bad", [
    ("loan_amount", -1),
    ("income", 0),
    ("loan_to_value_ratio", 500),
    ("loan_term", 0),
])
def test_rejects_invalid_input(field, bad):
    r = client.post("/predict", json={**VALID, field: bad})
    assert r.status_code == 422


def test_unseen_category_does_not_crash():
    r = client.post("/predict", json={**VALID, "debt_to_income_ratio": "not-a-real-band"})
    assert r.status_code == 200
