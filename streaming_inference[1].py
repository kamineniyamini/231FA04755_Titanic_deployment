import pandas as pd
import joblib
import time
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

stream_data = X_unseen.iloc[21:31]
model = joblib.load("/content/model.pkl")

for i, (index, passenger) in enumerate(stream_data.iterrows(), start=1):
    prediction = int(model.predict(passenger.to_frame().T)[0])

    print(
        "Record:", i,
        "| Actual:", int(y_unseen.loc[index]),
        "| Predicted:", prediction,
        "| Result:", "Survived" if prediction == 1 else "Did not survive"
    )

    time.sleep(1)

print("Streaming simulation completed.")
