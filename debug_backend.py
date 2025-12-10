import pandas as pd
import numpy as np
import json
import os
import traceback

def debug_transactions_endpoint():
    print("Starting debug process...")
    data_path = "data/processed/cleaned_transactions.csv"
    
    if not os.path.exists(data_path):
        print(f"ERROR: File not found at {data_path}")
        return

    print(f"Loading file: {data_path}")
    try:
        df = pd.read_csv(data_path)
        print("File loaded successfully.")
        print(f"Columns: {df.columns.tolist()}")
        print(f"Shape: {df.shape}")
        
        # Mimic exact main.py logic
        print("Applying main.py sanitization logic...")
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.where(pd.notnull(df), None)
        
        print("Converting to dict...")
        data = df.to_dict(orient="records")
        print(f"Converted {len(data)} records.")
        
        print("Attempting JSON serialization...")
        # This is where FastApi fails if pydantic/json can't handle a type
        json_output = json.dumps(data)
        
        print("SUCCESS: Data is valid JSON serializable.")
        
    except Exception as e:
        print("\n!!! ERROR DETECTED !!!")
        print(f"Exception Type: {type(e).__name__}")
        print(f"Exception Message: {str(e)}")
        print("\nFull Traceback:")
        traceback.print_exc()

if __name__ == "__main__":
    debug_transactions_endpoint()
