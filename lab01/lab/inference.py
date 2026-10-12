import joblib


def load_model(path="models/model.joblib"):
    return joblib.load(path)


def predict(model, data):
    features = [
        [
            data["sepal_length"],
            data["sepal_width"],
            data["petal_length"],
            data["petal_width"],
        ]
    ]
    prediction = model.predict(features)
    return str(prediction[0])
