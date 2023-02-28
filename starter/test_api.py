import pytest
from fastapi.testclient import TestClient
import json
import sys

from main import census_app

client = TestClient(census_app)


def test_get_path():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == ["Hello User!"]


def test_path_two():
    r = client.get("/MyName")
    assert r.status_code == 200
    assert r.json() == ["Welcome to this app, MyName"]

def test_post_less_than_fifty():
    data = {
        "age": 39,
        "workclass": "State-gov",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital-gain": 2174,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States",
    }
    headers = {"Content-Type": "application/json"}

    r = client.post("/predict", data=json.dumps(data), headers=headers)
    assert r.status_code == 200
    assert r.json()["prediction"] == " <=50K"


# @pytest.mark.skip(reason="Pass for now")
def test_post_more_than_fifty():
    data = {
        "age": 50,
        "workclass": "State-gov",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital-gain": 500000,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States"
    }
    headers = {"Content-Type": "application/json"}

    r = client.post("/predict", data=json.dumps(data), headers=headers)
    assert r.status_code == 200
    assert r.json()["prediction"] == " >50K"