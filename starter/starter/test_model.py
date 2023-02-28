import pandas as pd
import pytest
import os 

from .ml.data import process_data
from .ml.model import train_model, compute_model_metrics, inference, save_to_file
from sklearn.model_selection import train_test_split

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture()
def data():
    """
    Load the dataset used for the analysis
    Returns
    -------
    data: pd.DataFrame
    """
    # Add code to load in the data.
    data = pd.read_csv(os.path.join(root_dir, "data", "census.csv"))

    return data

def test_train_model(data):
    # Split dataset into training and testing set
    train, test = train_test_split(data, test_size=0.20, random_state=42)

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    target_var = "salary"

    X_train, y_train, encoder, lb= process_data(
        train, categorical_features=cat_features, label=target_var, training=True
    )

    model = train_model(X_train, y_train)
    preds = model.predict(X_train)
    assert preds.shape[0] == y_train.shape[0]

def test_inference(data):
    # Split dataset into training and testing set
    train, test = train_test_split(data, test_size=0.20, random_state=42)

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    target_var = "salary"

    X_train, y_train, encoder, lb= process_data(
        train, categorical_features=cat_features, label=target_var, training=True
    )

    X_test, y_test, encoder, lb= process_data(
        test, categorical_features=cat_features, encoder=encoder, lb=lb, label=target_var, training=False
    )

    model = train_model(X_train, y_train)
    preds = inference(model=model, X=X_test)
    assert preds.shape[0] == y_test.shape[0]

def test_save_model(data):
    # Split dataset into training and testing set
    train, test = train_test_split(data, test_size=0.20, random_state=42)

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    target_var = "salary"

    X_train, y_train, encoder, lb= process_data(
        train, categorical_features=cat_features, label=target_var, training=True
    )

    model = train_model(X_train, y_train)
    # Save the model.
    save_to_file(model.best_estimator_, os.path.join(root_dir, "model", "classifier"))

    assert os.path.isfile(root_dir + "/model/classifier.pkl")