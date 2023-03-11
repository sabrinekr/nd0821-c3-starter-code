import boto3
import botocore
import numpy as np
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder

BUCKET_NAME = "starter3"
KEY = "9e4671b0-8d3a-4451-b37a-6821f37b555d"

ACCESS_ID = "ASIATGYWG677TEA5XW6L"
ACCESS_KEY = "mBR/xcPJZ92b4v4DE6rG5zcOyNrsEBhoCaQnWsEI"
SESSION_TOKEN = "FwoGZXIvYXdzELL//////////wEaDDlnd/CWpL5b8a7rjyLVAWrt36cr6SLTfweMNBDpF5EWsSNn0z7MlnNZihu2nUA19Ii7EewaIfCPopUh6h27Pz7BfvSaQPgVXCZ8uocXwGkkiLd56aHbJazGBW8hUT5iOOKHlIiDLVuXBcvEIxNV90WzuSnwpUJs7wRok7y7vhnrujA9Y2j3NJ1Jpsn2UWa7dea3hiJv/f4twV3CbVW0svuIg4Un52kNPzUBREea08iZdT1fzOJGiG4uLIoskqIRvZuysfqiVqscTAZSjE1MOxqXw/sWOHc7VZOHWpKetWI9soqp0SjTpqagBjItJMFImkFKIX11BQQv26gGkgSR0GTHiNtHS7y091T/qm1NXsTmqIF5cs53Ss8I"

def process_data(
    X, categorical_features=[], label=None, training=True, encoder=None, lb=None
):
    """ Process the data used in the machine learning pipeline.

    Processes the data using one hot encoding for the categorical features and a
    label binarizer for the labels. This can be used in either training or
    inference/validation.

    Note: depending on the type of model used, you may want to add in functionality that
    scales the continuous data.

    Inputs
    ------
    X : pd.DataFrame
        Dataframe containing the features and label. Columns in `categorical_features`
    categorical_features: list[str]
        List containing the names of the categorical features (default=[])
    label : str
        Name of the label column in `X`. If None, then an empty array will be returned
        for y (default=None)
    training : bool
        Indicator if training mode or inference/validation mode.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder, only used if training=False.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer, only used if training=False.

    Returns
    -------
    X : np.array
        Processed data.
    y : np.array
        Processed labels if labeled=True, otherwise empty np.array.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained OneHotEncoder if training is True, otherwise returns the encoder passed
        in.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained LabelBinarizer if training is True, otherwise returns the binarizer
        passed in.
    """

    if label is not None:
        y = X[label]
        X = X.drop([label], axis=1)
    else:
        y = np.array([])

    X_categorical = X[categorical_features].values
    X_continuous = X.drop(*[categorical_features], axis=1)

    if training is True:
        encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
        lb = LabelBinarizer()
        X_categorical = encoder.fit_transform(X_categorical)
        y = lb.fit_transform(y.values).ravel()
    else:
        X_categorical = encoder.transform(X_categorical)
        try:
            y = lb.transform(y.values).ravel()
        # Catch the case where y is None because we're doing inference.
        except AttributeError:
            pass

    X = np.concatenate([X_continuous, X_categorical], axis=1)
    return X, y, encoder, lb

def load_data_s3(file_name, path):
    """Downloads data from s3 bucket to `path` loads it and returns it under a pandas.DataFrame format
    Args:
        path: (str) path to data (.csv) file
    Return:
        data: (pandas.DataFrame)
    """
    s3 = boto3.client('s3',
        region_name="us-east-1",
        aws_access_key_id=ACCESS_ID,
        aws_secret_access_key= ACCESS_KEY,
        aws_session_token=SESSION_TOKEN)

    s3.download_file(
        Bucket=BUCKET_NAME, Key=path, Filename=file_name
    )
    return path