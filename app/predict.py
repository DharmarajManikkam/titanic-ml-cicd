from fastapi import FastAPI
import joblib
import numpy as np
from pydantic import BaseModel

model = joblib.load("model.joblib")
app = FastAPI()

class Features(BaseModel):
    Pclass: int
    Sex: int
    Age: float
    SibSp: int
    Parch: int
    Fare: float
    Embarked: int

@app.post("/predict/")
def predict(features: Features):
    input_array = np.array([[features.Pclass, features.Sex, features.Age,
                             features.SibSp, features.Parch, features.Fare,
                             features.Embarked]])
    pred = model.predict(input_array)
    return {"survived": int(pred[0])}
