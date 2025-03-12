import os
import pandas as pd
import numpy as np
import joblib
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Suppress warnings
warnings.simplefilter(action='ignore', category=UserWarning)

#  Ensure "models" directory exists before saving models
model_dir = "C:/upi_fraud_past_basic/models/"
os.makedirs(model_dir, exist_ok=True)

#  Step 1: Load dataset
df = pd.read_csv("C:/upi_fraud_past_basic/data/upi_fraud_data.csv")

#  Step 2: Feature Selection
features = ["User_ID", "Transaction_Amount", "Transaction_Type", "Location", "Device_Type"]
target = "Is_Fraudulent"

X = df[features]
y = df[target]

#  Step 3: Convert categorical variables into numerical values
X = pd.get_dummies(X, drop_first=True)

#  Step 4: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#  Step 5: Standardizing numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#  Save the scaler for later use in prediction
joblib.dump(scaler, os.path.join(model_dir, "scaler.pkl"))
print("Scaler saved as 'scaler.pkl'.")

#  Step 6: Train Models
# Model 1: Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)
y_pred_rf = rf_model.predict(X_test_scaled)

#  Model 2: XGBoost
xgb_model = XGBClassifier(eval_metric="logloss", random_state=42)
xgb_model.fit(X_train_scaled, y_train)
y_pred_xgb = xgb_model.predict(X_test_scaled)

#  Model 3: Logistic Regression
lr_model = LogisticRegression()
lr_model.fit(X_train_scaled, y_train)
y_pred_lr = lr_model.predict(X_test_scaled)

#  Step 7: Save trained models
joblib.dump(rf_model, os.path.join(model_dir, "random_forest.pkl"))
joblib.dump(xgb_model, os.path.join(model_dir, "xgboost.pkl"))
joblib.dump(lr_model, os.path.join(model_dir, "logistic_regression.pkl"))
joblib.dump(X_train.columns, "C:/upi_fraud_past_basic/models/trained_columns.pkl")
print(" Models saved successfully!")

#  Step 8: Function to print evaluation metrics
def print_metrics(model_name, y_true, y_pred):
    print(f"\n📌 {model_name} Performance:")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall: {recall_score(y_true, y_pred):.4f}")
    print(f"F1-Score: {f1_score(y_true, y_pred):.4f}")

#  Step 9: Print model performances
print_metrics("Random Forest", y_test, y_pred_rf)
print_metrics("XGBoost", y_test, y_pred_xgb)
print_metrics("Logistic Regression", y_test, y_pred_lr)
