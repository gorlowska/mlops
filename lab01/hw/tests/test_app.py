import pytest
from app import app
from fastapi.testclient import TestClient

client = TestClient(app)


def test_predict_returns_valid_json():
    response = client.post("/predict", json={"text": "I love this movie"})
    assert response.status_code == 200
    assert response.json() == {"prediction": "positive"}


@pytest.mark.parametrize("payload", [{"text": ""}, {"text": 123}, {}])
def test_predict_invalid_input_returns_error_json(payload):
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    assert "detail" in response.json()
