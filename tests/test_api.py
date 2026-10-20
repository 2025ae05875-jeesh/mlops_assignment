"""API contract, validation and error handling (FastAPI TestClient)."""

import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from tests.conftest import VALID_PATIENT


@pytest.fixture()
def client(model_file):
    with TestClient(create_app(str(model_file))) as c:  # runs lifespan
        yield c


@pytest.fixture()
def client_without_model(tmp_path):
    with TestClient(create_app(str(tmp_path / "missing.joblib"))) as c:
        yield c


def test_health_and_ready(client):
    assert client.get("/health").json() == {"status": "ok"}
    r = client.get("/ready")
    assert r.status_code == 200 and r.json()["model_name"] == "test_lr"


def test_predict_valid_payload(client):
    r = client.post("/predict", json=VALID_PATIENT)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in (0, 1)
    assert 0 <= body["probability_disease"] <= 1
    assert 0.5 <= body["confidence"] <= 1
    assert body["model_name"] == "test_lr"
    assert "x-request-id" in r.headers


def test_predict_missing_field_returns_422(client):
    payload = {k: v for k, v in VALID_PATIENT.items() if k != "chol"}
    r = client.post("/predict", json=payload)
    assert r.status_code == 422
    assert any(e["field"] == "chol" for e in r.json()["errors"])


def test_predict_out_of_range_and_wrong_type_return_422(client):
    too_old = client.post("/predict", json={**VALID_PATIENT, "age": 500})
    wrong_type = client.post("/predict", json={**VALID_PATIENT, "cp": "x"})
    assert too_old.status_code == 422
    assert wrong_type.status_code == 422


def test_predict_unknown_field_returns_422(client):
    r = client.post("/predict", json={**VALID_PATIENT, "patient_name": "X"})
    assert r.status_code == 422


def test_predict_malformed_json_returns_422(client):
    r = client.post("/predict", content=b"{not json",
                    headers={"content-type": "application/json"})
    assert r.status_code == 422


def test_metrics_exposed_and_counting(client):
    client.post("/predict", json=VALID_PATIENT)
    client.post("/predict", json={"age": 1})
    text = client.get("/metrics").text
    assert "heart_api_requests_total" in text
    assert "heart_api_validation_errors_total" in text
    assert "heart_api_predictions_total" in text


def test_service_reports_not_ready_without_model(client_without_model):
    assert client_without_model.get("/health").status_code == 200
    assert client_without_model.get("/ready").status_code == 503
    assert client_without_model.post("/predict", json=VALID_PATIENT).status_code == 503
