import pandas as pd
import numpy as np
import joblib
import warnings
import os

# Suppress warnings
warnings.simplefilter(action='ignore', category=UserWarning)

#  Load the trained scaler and models
try:
    scaler = joblib.load("C:/fraud_detection_using_machine_learning/models/scaler.pkl")
    rf_model = joblib.load("C:/fraud_detection_using_machine_learning/models/random_forest.pkl")
    xgb_model = joblib.load("C:/fraud_detection_using_machine_learning/models/xgboost.pkl")
    lr_model = joblib.load("C:/fraud_detection_using_machine_learning/models/logistic_regression.pkl")
    print(" Scaler and models loaded successfully!")
except FileNotFoundError as e:
    print(f" ERROR: {e}")
    exit()

#  Load new transaction data
data_path = "C:/fraud_detection_using_machine_learning/data/new_transactions.csv"

try:
    df = pd.read_csv(data_path)
    print(" New transactions loaded successfully!")
except FileNotFoundError:
    print(f" ERROR: {data_path} not found.")
    exit()

# Define feature names exactly as in training
features = ["User_ID", "Transaction_Amount", "Transaction_Type", "Location", "Device_Type"]

# Ensure only the required columns are used
if not set(features).issubset(df.columns):
    missing_features = list(set(features) - set(df.columns))
    print(f" ERROR: Missing columns in new_transactions.csv -> {missing_features}")
    exit()

X_new = df[features]

# Apply the same encoding as in training
X_new = pd.get_dummies(X_new, drop_first=True)

#  Load training feature names (ensures correct order & prevents missing column errors)
trained_columns = joblib.load("C:/fraud_detection_using_machine_learning/models/trained_columns.pkl")

# Add missing columns with 0 values
missing_cols = set(trained_columns) - set(X_new.columns)
for col in missing_cols:
    X_new[col] = 0

# Ensure column order matches training
X_new = X_new[trained_columns]

# Scale the new transaction data
X_new_scaled = scaler.transform(X_new)

#  Make predictions (using the best model - RF)
y_pred = rf_model.predict(X_new_scaled)

#  Add predictions to the dataframe
df["Is_Fraudulent_Predicted"] = y_pred

# Ensure the output directory exists
output_dir = "C:/fraud_detection_using_machine_learning/results/"
os.makedirs(output_dir, exist_ok=True)  # Creates the directory if it doesn't exist

#  Save the predictions (Only saving Transaction_ID & Prediction for evaluation)
output_path = os.path.join(output_dir, "predictions.csv")
df[["Transaction_ID", "Is_Fraudulent_Predicted"]].to_csv(output_path, index=False)

print(f"Predictions saved successfully to {output_path}!")