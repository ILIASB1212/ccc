from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from starlette import status

import joblib
import pandas as pd


try:
    model = joblib.load("xgboost_model.joblib")
    preprosses = joblib.load("preprocessor.joblib")
    print("Model and preprocessor loaded successfully.")
except FileNotFoundError:
    print("Error: Model or preprocessor files not found. Please ensure 'xgboost_model.joblib' and 'preprocessor.joblib' are in the same directory.")
    raise RuntimeError("Required model files not found. Cannot start the API.")


app = FastAPI(
    title="Customer Churn Prediction API",
    description="A REST API for predicting customer churn based on various features."
)


class CustomerData(BaseModel):
    CreditScore: int = Field(gt=-1, lt=1001)
    Geography: str = Field(min_length=2, max_length=20)
    Gender: str = Field(min_length=2, max_length=10)
    Age: int = Field(gt=17, lt=101)
    Tenure: int = Field(ge=0, le=11)
    Balance: float = Field(ge=0)  
    NumOfProducts: int = Field(ge=0, le=5)
    HasCrCard: int = Field(ge=0, le=1) 
    IsActiveMember: int = Field(ge=0, le=1) 
    EstimatedSalary: float = Field(ge=0) 

INPUT_COLUMNS_ORDER = [ 
    'CreditScore', 'Geography', 'Gender', 'Age', 'Tenure',
    'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary'
]

@app.post("/predict", status_code=status.HTTP_200_OK)
async def predict(client_d: CustomerData):
    data = client_d.model_dump()
    input_df = pd.DataFrame([data], columns=INPUT_COLUMNS_ORDER)

    try:
        transformed = preprosses.transform(input_df)
        predict_class = model.predict(transformed)[0] 
        predict_probas = model.predict_proba(transformed)[0] 

        result_status = "churn" if predict_class == 1 else "stayed" 
        
        churn_probability_percent = predict_probas[1] * 100 

        print(f"The client will {result_status}")
        print(f"Received data: {data}")
        print(f"Transformed data shape: {transformed.shape}")
        print(f"Raw prediction: {predict_class}, Churn Probability: {churn_probability_percent:.2f}%")
        print(f"The client is predicted to {result_status} with {churn_probability_percent:.2f}% probability.")

        return {
            "prediction_message": f"The customer is predicted to {result_status}.",
            "churn_probability": f"{churn_probability_percent:.2f}%",  }

    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Prediction error: {str(e)}")


@app.get("/", status_code=status.HTTP_200_OK)
async def root():
    return {"message": "Welcome to the Churn Prediction API! Send a POST request to /predict for predictions."}
