import inference
from api.models.sentiment import PredictRequest, PredictResponse
from fastapi import FastAPI

app = FastAPI()

sentence_transformer = inference.load_sentence_transformer()
classifier = inference.load_classifier()


@app.post("/predict")
def predict(request: PredictRequest) -> PredictResponse:
    prediction = inference.predict(request.text, sentence_transformer, classifier)
    return PredictResponse(prediction=prediction)
