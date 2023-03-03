import dill as pickle
import pandas as pd

from fastapi import FastAPI, HTTPException, Request
from fastapi.templating import Jinja2Templates

# Import Union since our Item object will have tags that can be
# strings or a list.
# from typing import Union
import uvicorn
# BaseModel from Pydantic is used to define data objects.
from pydantic import BaseModel, Field

from starter.demo.ml.data import process_data
from starter.demo.ml.model import inference, load_from_file

import os

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# DVC set-up for Render
if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    os.system("dvc config core.hardlink_lock true")
    if os.system("dvc pull  -r storage") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")

# Load the preprocessors and the classifier
encoder = load_from_file(os.path.join(root_dir, "starter/starter/model", "encoder"))
classifier = load_from_file(os.path.join(root_dir, "starter/starter/model", "classifier"))
lb = load_from_file(os.path.join(root_dir, "starter/starter/model", "lb"))

class ClassifierFeatureIn(BaseModel):
    age: int = Field(..., example=39)
    workclass: str = Field(..., example="State-gov")
    fnlgt: int = Field(..., example=77516)
    education: str = Field(..., example="Bachelors")
    education_num: int = Field(..., example=13, alias="education-num")
    marital_status: str = Field(..., example="Never-married",
                                alias="marital-status")
    occupation: str = Field(..., example="Adm-clerical")
    relationship: str = Field(..., example="Not-in-family")
    race: str = Field(..., example="White")
    sex: str = Field(..., example="Male")
    capital_gain: int = Field(..., example=2174, alias="capital-gain")
    capital_loss: int = Field(..., example=0, alias="capital-loss")
    hours_per_week: int = Field(..., example=40, alias="hours-per-week")
    native_country: str = Field(..., example="United-States",
                                alias="native-country")


# declare fastapi app
census_app = FastAPI()

# Define a GET for greetings.
@census_app.get("/")
async def greet_user():
    return {"Hello User!"}


# Greet the user with his/her name
@census_app.get("/{name}")
async def get_name(name: str):
    return {f"Welcome to this app, {name}"}


# response_model=ClassifierOut, status_code=200
@census_app.post("/predict")
def predict(input_data: ClassifierFeatureIn):
    data = pd.DataFrame.from_dict([input_data.dict(by_alias=True)])

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

    # Preprocess the data
    X, _, _, _ = process_data(
        data, categorical_features=cat_features, encoder=encoder, lb=lb, training=False
    )

    # Predict salary
    preds = inference(classifier, X)
    preds = lb.inverse_transform(preds)

    return {
        "prediction": preds[0]
    }

if __name__ == "__main__":
    uvicorn.run(census_app, host="127.0.0.1", port=8000)