import requests
import json

import pandas as pd

data = pd.read_csv("starter/data/census.csv")

X = data.iloc[0]
y = X.pop('salary')

r = requests.post('https://starter-pty1.onrender.com/api', data=json.dumps(X.to_dict()))

print(f"status code: {r.status_code}")
print(f"prediction: {r.json()}, \t ground truth: {y}")