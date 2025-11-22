# tests/test_api.py
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_predict_ok():
    payload = {
        "age": 35,
        "gender": "M",
        "marital_status": "single",
        "dependents": 1,
        "monthly_income": 3500.0,
        "employment_type": "permanent",
        "employment_months": 48,
        "requested_amount": 25000.0,
        "loan_term_months": 36,
        "interest_rate": 18.0,
        "installment": 950.0,
        "debt_to_income": 0.45,
        "num_open_loans": 2,
        "num_credit_cards": 2,
        "has_mortgage": 0,
        "channel": "web",
        "region": "capital",
    }
    resp = client.post("/predict/", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert "default_probability" in body
    assert "default_class" in body
