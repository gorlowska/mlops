import joblib
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier


def load_data():
    return load_iris()


def train_model(data):
    X = data.data
    y = data.target_names[data.target]
    model = DecisionTreeClassifier()
    model.fit(X, y)
    return model


def save_model(model, path="models/model.joblib"):
    joblib.dump(model, path)


if __name__ == "__main__":
    data = load_data()
    model = train_model(data)
    save_model(model)
