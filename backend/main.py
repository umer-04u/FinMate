from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import pickle
import os
import uvicorn
from typing import List, Optional

app = FastAPI(title="Budget Analysis AI API")

# --- CORS Configuration ---
from fastapi.middleware.cors import CORSMiddleware

origins = [
    "http://localhost:5173", # Vite default
    "http://localhost:3000",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Load Models ---
# Ideally these should be loaded once at startup
MODELS = {}

def load_models():
    model_dir = "../models"
    try:
        with open(f"{model_dir}/anomaly_model.pkl", "rb") as f:
            MODELS["anomaly"] = pickle.load(f)
        with open(f"{model_dir}/spending_forecaster.pkl", "rb") as f:
            MODELS["forecast"] = pickle.load(f)
        with open(f"{model_dir}/merchant_categorizer.pkl", "rb") as f:
            MODELS["categorizer"] = pickle.load(f)
        with open(f"{model_dir}/merchant_vectorizer.pkl", "rb") as f:
            MODELS["vectorizer"] = pickle.load(f)
        with open(f"{model_dir}/category_encoder.pkl", "rb") as f:
            MODELS["encoder"] = pickle.load(f)
        print("All models loaded successfully.")
    except Exception as e:
        print(f"Error loading models: {e}")
        # In production this might be fatal, but for dev we continue
        pass

@app.on_event("startup")
async def startup_event():
    load_models()

# --- Pydantic Models for Input ---
class TransactionInput(BaseModel):
    amount: float
    category: str
    date: str # YYYY-MM-DD
    merchant: Optional[str] = None

class CategorizeInput(BaseModel):
    merchant: str

# --- Endpoints ---

@app.get("/")
def read_root():
    return {"message": "Budget Analysis AI API is running"}

@app.post("/analyze/anomaly")
def check_anomaly(transaction: TransactionInput):
    """
    Checks if a single transaction is an anomaly.
    """
    if "anomaly" not in MODELS:
        raise HTTPException(status_code=503, detail="Anomaly model not loaded")
    
    # Preprocess Input
    # We need to match features trained on: Amount, Category_Encoded, DayOfWeek, IsWeekend
    try:
        # Encode Category
        if "encoder" in MODELS:
            try:
                cat_encoded = MODELS["encoder"].transform([transaction.category])[0]
            except:
                cat_encoded = -1 # Unknown category
        else:
            cat_encoded = 0
            
        dt = pd.to_datetime(transaction.date)
        day_of_week = dt.dayofweek
        is_weekend = 1 if day_of_week >= 5 else 0
        
        # Prepare Feature Vector
        features = [[transaction.amount, cat_encoded, day_of_week, is_weekend]]
        
        # Predict
        prediction = MODELS["anomaly"].predict(features)[0]
        # -1 is anomaly, 1 is normal
        is_anomaly = True if prediction == -1 else False
        
        return {
            "is_anomaly": is_anomaly,
            "confidence": "High" # Isolation forest decision function could vary
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/spending")
def predict_spending(prev_month_spend: float):
    """
    Predicts next month's spending based on previous month.
    """
    if "forecast" not in MODELS:
        raise HTTPException(status_code=503, detail="Forecast model not loaded")
    
    try:
        # Our simple model trained on MonthIndex and PrevMonthSpend
        # For this API, let's just use a dummy MonthIndex (e.g., 12) or try to adapt
        # Ideally we'd pass the full history. 
        # Since we trained on MonthIndex + Prev, let's assume next month is index 25 (arbitrary future)
        # In a real app we'd track time.
        
        features = [[25, prev_month_spend]] 
        prediction = MODELS["forecast"].predict(features)[0]
        
        return {"predicted_spend": round(prediction, 2)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/categorize")
def categorize_merchant(input_data: CategorizeInput):
    """
    Predicts category for a given merchant name.
    """
    if "categorizer" not in MODELS or "vectorizer" not in MODELS:
        raise HTTPException(status_code=503, detail="Categorization model not loaded")
    
    try:
        # Vectorize
        vectorized_text = MODELS["vectorizer"].transform([input_data.merchant]).toarray()
        
        # Predict
        pred_encoded = MODELS["categorizer"].predict(vectorized_text)[0]
        
        # Decode
        pred_label = MODELS["encoder"].inverse_transform([pred_encoded])[0]
        
        return {"merchant": input_data.merchant, "category": pred_label}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/transactions")
def get_transactions():
    """
    Returns transaction history for the dashboard.
    """
    try:
        # Load from the processed CSV used by the original app
        # Adjust path as needed relative to backend/main.py
        data_path = "../data/processed/cleaned_transactions.csv"
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
            
            # Simple, robust way to handle NaN/Inf for JSON compliance:
            # Convert to dict first, then recursively clean.
            records = df.to_dict(orient="records")
            
            def clean_nan(obj):
                if isinstance(obj, float):
                    if np.isnan(obj) or np.isinf(obj):
                        return None
                elif isinstance(obj, dict):
                    return {k: clean_nan(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [clean_nan(i) for i in obj]
                return obj
                
            cleaned_records = [clean_nan(record) for record in records]
            return cleaned_records
        else:
            return []
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/transactions")
def add_transaction(transaction: TransactionInput):
    """
    Adds a new transaction to the CSV file.
    """
    try:
        data_path = "../data/processed/cleaned_transactions.csv"
        
        # Create a new record
        new_record = {
            "Date": transaction.date,
            "Description": transaction.merchant if transaction.merchant else "Unknown",
            "Amount": transaction.amount,
            "Category": transaction.category,
            "Is_Expense": True, # Assume expense for now
            "Month": pd.to_datetime(transaction.date).strftime("%Y-%m")
        }
        
        # Append to CSV
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
            # Use pd.concat properly
            df = pd.concat([df, pd.DataFrame([new_record])], ignore_index=True)
            df.to_csv(data_path, index=False)
        else:
            df = pd.DataFrame([new_record])
            df.to_csv(data_path, index=False)
            
        return {"message": "Transaction added successfully", "transaction": new_record}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class ChatInput(BaseModel):
    query: str

@app.post("/chat")
def chat_bot(input_data: ChatInput):
    """
    Simple chatbot endpoint. 
    In future, use GEMINI_API_KEY from .env to call Google Gemini API.
    """
    try:
        # Placeholder logic
        # if os.getenv("GEMINI_API_KEY"): ... call API ...
        
        return {
            "response": f"I received your query: '{input_data.query}'. To function fully, I need a valid API key configured in the backend."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
