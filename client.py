import requests

url = "http://127.0.0.1:8000/predict"

passenger = {
    "Pclass": 3,
    "Sex": "female",
    "Age": 25,
    "SibSp": 0,
    "Parch": 0,
    "Fare": 7.25,
    "Embarked": "S"
}

response = requests.post(url, json=passenger)

print("Status Code:", response.status_code)
print("Prediction:", response.json())
