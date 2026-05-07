from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import numpy as np

app = FastAPI(title="Customer Churn Prediction API")

# Load model and feature list
try:
    model = joblib.load('best_model.pkl')
    features = joblib.load('model_features.pkl')
except:
    model = None
    features = None

class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: str  # Kept as string to match initial CSV state

def preprocess_input(data: dict):
    df = pd.DataFrame([data])
    
    # Simulate load_and_clean_data steps
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(0, inplace=True)
    
    # Mapping for label encoded variables (0 or 1 usually depends on the fit, 
    # but normally le.fit_transform does alphabetical. 
    # 'gender': Female=0, Male=1
    # 'Partner': No=0, Yes=1
    # 'Dependents': No=0, Yes=1
    # 'PhoneService': No=0, Yes=1
    # 'PaperlessBilling': No=0, Yes=1
    
    mapping = {'No': 0, 'Yes': 1, 'Female': 0, 'Male': 1}
    binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    for col in binary_cols:
        df[col] = df[col].map(mapping).fillna(0)
        
    # Feature engineering
    bins = [0, 12, 24, 36, 48, 60, 72]
    labels = [1, 2, 3, 4, 5, 6]
    df['tenure_group'] = pd.cut(df['tenure'], bins=bins, labels=labels, right=True, include_lowest=True)
    df['tenure_group'] = df['tenure_group'].astype(int)
    
    df['avg_charge_per_month'] = np.where(df['tenure'] > 0, df['TotalCharges'] / df['tenure'], df['MonthlyCharges'])
    
    services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    df['services_count'] = df[services].apply(lambda x: (x == 'Yes').sum(), axis=1)
    df['has_tech_support'] = (df['TechSupport'] == 'Yes').astype(int)
    
    # Encode categoricals (dummy variables)
    cat_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
                'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 
                'Contract', 'PaymentMethod']
    
    df = pd.get_dummies(df, columns=cat_cols)
    
    # Ensure all column names are strings and clean
    df.columns = [str(c).replace(' ', '_').replace('(', '').replace(')', '') for c in df.columns]
    
    # Align with training features
    for f in features:
        if f not in df.columns:
            df[f] = False # or 0 depending on the type
            
    # Keep only the features used in training, in the right order
    df = df[features]
    
    # Convert booleans to int for some models just in case
    df = df.astype(float)
    
    return df

@app.post("/predict")
def predict_churn(customer: CustomerData):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")
        
    data_dict = customer.dict()
    df_processed = preprocess_input(data_dict)
    
    probability = model.predict_proba(df_processed)[0][1]
    prediction = int(probability > 0.5)
    
    return {
        "churn_probability": float(probability),
        "churn_prediction": prediction,
        "score": round(probability * 100, 2)
    }

@app.get("/")
def health_check():
    return {"status": "ok", "model_loaded": model is not None}
