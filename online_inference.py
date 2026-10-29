import pandas as pd
import joblib
from sklearn.model_selection import train_test_split

df = pd.read_csv("/content/train.csv")
df = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])

X = df.drop(columns=["Survived"])
y = df["Survived"]

_, X_temp, _, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

X_unseen, _, y_unseen, _ = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

model = joblib.load("/content/model.pkl")
passenger = X_unseen.iloc[[0]]
prediction = int(model.predict(passenger)[0])

print("Passenger Details:")
print(passenger)
print("Actual Survival:", int(y_unseen.iloc[0]))
print("Predicted Survival:", prediction)
print("Result:", "Survived" if prediction == 1 else "Did not survive")
