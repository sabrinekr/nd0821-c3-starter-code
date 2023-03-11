import json
import requests
import argparse


def test_api_live_get_predictions_inf1(args):
    """ Test Fast API predict route with a '<=50K' salary prediction result """

    app_url = args.url + "/predict"
    print(f"test_api_live_get_predictions_inf1 -> {app_url}...")

    expected_res = "Predicts ['<=50K']"

    test_data = {
        "age": 4,
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

    response = {}

    r = requests.post(app_url, data=json.dumps(test_data), headers=headers)
    response['prediction'] = r.json()["prediction"][: len(expected_res)]
    response['status'] = r.status_code
    assert r.status_code == 200
    return response


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Census Bureau Heroku App Predictions Test CLI"
    )

    parser.add_argument(
        "url",
        type=str,
        help="url[:port] of the app to test inferences (e.g. http://127.0.0.1:8000)",
    )

    args = parser.parse_args()

    print(f"testing live app prediction for {args.url}...")

    # Call live testing function
    print("test_api_live_get_predictions_inf1 ...")
    res = test_api_live_get_predictions_inf1(args)
    print(res)



