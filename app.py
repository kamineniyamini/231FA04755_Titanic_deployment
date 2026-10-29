from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
import joblib

app = FastAPI(title="Titanic Survival Prediction API")

model = joblib.load("/content/model.pkl")

class Passenger(BaseModel):
    Pclass: int = Field(ge=1, le=3)
    Sex: str
    Age: float = Field(ge=0, le=120)
    SibSp: int = Field(ge=0)
    Parch: int = Field(ge=0)
    Fare: float = Field(ge=0)
    Embarked: str

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
def predict(passenger: Passenger):
    data = pd.DataFrame([passenger.model_dump()])
    prediction = int(model.predict(data)[0])

    return {
        "predicted_survival": prediction,
        "result": "Survived" if prediction == 1 else "Did not survive"
    }
