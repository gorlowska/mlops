import inference
import pytest

sentence_transformer = inference.load_sentence_transformer()
classifier = inference.load_classifier()


def test_models_load():
    assert sentence_transformer is not None
    assert classifier is not None


@pytest.mark.parametrize(
    "text, expected",
    [
        ("I love this movie", "positive"),
        ("This food is terrible", "negative"),
        ("Today is Monday", "neutral"),
    ],
)
def test_predict(text, expected):
    assert inference.predict(text, sentence_transformer, classifier) == expected
