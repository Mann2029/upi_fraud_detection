import pandas as pd
import numpy as np
import joblib
import os
import time
from datetime import datetime

# Define paths
DATA_FOLDER = "C:/fraud_detection_using_machine_learning/data/new_batches/"
RESULTS_FOLDER = "C:/fraud_detection_using_machine_learning/results/"
MODELS_FOLDER = "C:/fraud_detection_using_machine_learning/models/"

# Ensure required folders exist
os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Load trained models & scaler
try:
    scaler = joblib.load(os.path.join(MODELS_FOLDER, "scaler.pkl"))
    rf_model = joblib.load(os.path.join(MODELS_FOLDER, "random_forest.pkl"))
    xgb_model = joblib.load(os.path.join(MODELS_FOLDER, "xgboost.pkl"))
    lr_model = joblib.load(os.path.join(MODELS_FOLDER, "logistic_regression.pkl"))
    trained_columns = joblib.load(os.path.join(MODELS_FOLDER, "trained_columns.pkl"))
    print("Models and scaler loaded successfully!")
except FileNotFoundError as e:
    print(f"ERROR: Model file not found! {e}")
    exit()

# Get list of CSV files in the data folder
batch_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith(".csv")]

# Check if any transaction files exist
if not batch_files:
    print(f"ERROR: No CSV files found in '{DATA_FOLDER}'. Add files and try again.")
    exit()

# Process each transaction batch file
for file in batch_files:
    file_path = os.path.join(DATA_FOLDER, file)
    
    try:
        # Load the new transaction batch
        df = pd.read_csv(file_path)
        print(f"\nProcessing file: {file}")
        print(f" - Loaded {df.shape[0]} transactions")

        # Define required features
        features = ["User_ID", "Transaction_Amount", "Transaction_Type", "Location", "Device_Type"]

        # Check if required features exist
        missing_cols = [col for col in features if col not in df.columns]
        if missing_cols:
            print(f"ERROR: Missing columns in {file} -> {missing_cols}")
            continue  # Skip this file

        # Preprocess the data
        X_new = df[features]
        X_new = pd.get_dummies(X_new, drop_first=True)  # Encode categorical variables

        # Add missing columns from training
        for col in set(trained_columns) - set(X_new.columns):
            X_new[col] = 0

        # Ensure column order matches training
        X_new = X_new[trained_columns]

        # Check if X_new has any rows
        if X_new.shape[0] == 0:
            print(f"WARNING: No valid rows to process in file {file}. Skipping.")
            continue

        # Scale data
        X_new_scaled = scaler.transform(X_new)

        # Make fraud predictions
        df["Fraud_Prediction_RF"] = rf_model.predict(X_new_scaled)
        df["Fraud_Prediction_XGB"] = xgb_model.predict(X_new_scaled)
        df["Fraud_Prediction_LR"] = lr_model.predict(X_new_scaled)

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"predictions_{timestamp}.csv"
        df.to_csv(os.path.join(RESULTS_FOLDER, output_file), index=False)
        print(f" - Predictions saved: {output_file}")

    except Exception as e:
        print(f"ERROR processing {file}: {e}")

print("\n✅ Batch fraud detection completed for all files!")
