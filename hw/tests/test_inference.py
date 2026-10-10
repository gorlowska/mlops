import inference
import pytest


@pytest.fixture(scope="module")
def sentence_transformer():
    return inference.load_sentence_transformer()


@pytest.fixture(scope="module")
def classifier():
    return inference.load_classifier()


def test_sentence_transformer_loads(sentence_transformer):
    assert sentence_transformer.get_sentence_embedding_dimension() == 384


def test_classifier_loads(classifier):
    assert hasattr(classifier, "predict")


@pytest.mark.parametrize(
    "text, expected",
    [
        ("I love this movie", "positive"),
        ("This food is terrible", "negative"),
        ("Today is Monday", "neutral"),
    ],
)
def test_predict_returns_expected_sentiment(
    text, expected, sentence_transformer, classifier
):
    assert inference.predict(text, sentence_transformer, classifier) == expected
