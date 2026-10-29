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

batch_data = X_unseen.iloc[1:21]
model = joblib.load("/content/model.pkl")
predictions = model.predict(batch_data)

results = batch_data.copy()
results["Actual_Survived"] = y_unseen.loc[batch_data.index]
results["Predicted_Survived"] = predictions

results.to_csv("/content/batch_predictions.csv", index=False)

print(results)
print("Total predictions:", len(results))
print("Saved batch_predictions.csv")
