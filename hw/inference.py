import joblib
from cleantext import clean
from sentence_transformers import SentenceTransformer

SENTIMENT_LABELS = {0: "negative", 1: "neutral", 2: "positive"}


def load_sentence_transformer(
    path: str = "models/sentence_transformer.model",
) -> SentenceTransformer:
    return SentenceTransformer(path)


def load_classifier(path: str = "models/classifier.joblib"):
    return joblib.load(path)


def predict(text: str, sentence_transformer: SentenceTransformer, classifier) -> str:
    cleaned_text = clean(text)
    embeddings = sentence_transformer.encode([cleaned_text])
    predicted_class = classifier.predict(embeddings)[0]
    return SENTIMENT_LABELS[int(predicted_class)]
