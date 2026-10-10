from api.models.sentiment import PredictRequest, PredictResponse
from fastapi import FastAPI

app = FastAPI()


@app.post("/predict")
def predict(request: PredictRequest) -> PredictResponse:
    return PredictResponse(prediction="positive")
