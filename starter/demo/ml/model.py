import os
import pickle
import pandas as pd

from .data import process_data
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

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


def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    # instantiate the model
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)

    params = {"n_estimators": [100, 200, 300], "max_depth": [5, 10]}

    model_cv = GridSearchCV(
        rf,
        param_grid=params,
        scoring="accuracy",
        cv=5,
        n_jobs=-1,
    )

    # fit the model on the training set
    model_cv.fit(X_train, y_train)

    model = model_cv
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)

    return preds


def save_to_file(instance, filename):
    """
    Save an instance from a file.
    ----------
    instance: instance to save
    filename: name of the file to save
    Returns
    -------
    """
    with open(f"{filename}.pkl", "wb") as f:
        pickle.dump(instance, f)


def load_from_file(filename):
    """
    Loads an instance from a file.
    ----------
    filename: name of the file to load
    Returns
    -------
    """
    with open(f"{filename}.pkl", "rb") as f:
        model = pickle.load(f)
    return model


def slice_metrics_perfomance(
        df,
        feature,
        model,
        encoder,
        binarizer):
    """
    Computes model metrics based on data slices
    Inputs
    ------
    df : pd.DataFrame
         Dataframe containing the cleaned data
    category : str
         Dataframe column to slice
    rf_model: 
         Random forest model used to perform prediction
    encoder: OneHotEncoder
         Trained OneHotEncoder
    binarizer: LabelBinarizer
        Trained LabelBinarizer
     Returns
     -------
     predictions : dict
          Dictionary containing the predictions for each category feature
    """
    cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country"]
    
    target_var = "salary"

    predictions = {}
    with open(os.path.join(os.getcwd(), "slice_output.txt"), "w") as fp:
        for feat in df[feature].unique():
            fp.write(f"Value = {feat}\n")
            df_temp = df[df[feature] == feat]
            X, y, _, _ = process_data(
                df_temp, 
                categorical_features=cat_features, 
                label="salary", 
                training=False, 
                encoder = encoder, 
                lb = binarizer)

            predict_values = inference(model, X)
            precision, recall, fbeta = compute_model_metrics(y, predict_values)
            metrics = f"precision: {precision} \t recall: {recall} \t f1_score: {fbeta} \n"
            fp.write(metrics)

            predictions[feat] = {
                'precision': precision,
                'recall': recall,
                'fbeta': fbeta,
                'rows': len(df_temp)}
    return predictions

