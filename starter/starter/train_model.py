# Script to train machine learning model.
import os
import pandas as pd

from ml.model import train_model, compute_model_metrics, inference, save_to_file
from ml.data import process_data
from sklearn.model_selection import train_test_split

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Add code to load in the data.
data = pd.read_csv(os.path.join(root, "data", "census.csv"))

train, test = train_test_split(data, test_size=0.20)

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

X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label=target_var, training=True
)

X_test, y_test, encoder, lb = process_data(
    test,
    categorical_features=cat_features,
    encoder=encoder,
    lb=lb,
    label=target_var,
    training=False,
)

# Train and save a model.
model = train_model(X_train, y_train)

# Test the model.
preds = inference(model=model, X=X_test)

precision, recall, fbeta = compute_model_metrics(y=y_test, preds=preds)

# Save the model.
save_to_file(model.best_estimator_, os.path.join(root, "model", "classifier"))
