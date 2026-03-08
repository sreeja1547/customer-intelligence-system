from fastapi import FastAPI
import joblib
import pandas as pd

app= FastAPI()
pipeline = joblib.load("model/customer_model.pkl")

from pydantic import BaseModel

class ChurnInput(BaseModel):
    SeniorCitizen: int
    tenure: int
    MonthlyCharges: float
    TotalCharges: float
    gender: str
    Partner: str
    Dependents: str
    PhoneService: str
    InternetService: str
    Contract: str
    MultipleLines: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    PaperlessBilling: str
    PaymentMethod: str
@app.post("/predict_churn")
def predict_churn(data: ChurnInput):

    input_df = pd.DataFrame([data.dict()])
    prediction = pipeline.predict(input_df)[0]

    return {"Churn Prediction": prediction}

