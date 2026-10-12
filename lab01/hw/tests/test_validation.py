import pytest
from api.models.sentiment import PredictRequest, PredictResponse
from pydantic import ValidationError


def test_request_accepts_non_empty_text():
    request = PredictRequest(text="Great lecture")
    assert request.text == "Great lecture"


def test_request_rejects_empty_text():
    with pytest.raises(ValidationError):
        PredictRequest(text="")


def test_request_rejects_non_string_text():
    with pytest.raises(ValidationError):
        PredictRequest(text=123)


def test_response_requires_string_prediction():
    with pytest.raises(ValidationError):
        PredictResponse(prediction=None)
