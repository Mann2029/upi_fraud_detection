import pandas as pd
from sklearn.model_selection import train_test_split

# Load preprocessed data
df = pd.read_csv("C:/upi_fraud_past_basic/data/upi_fraud_data.csv")  # Ensure preprocessing script saves this

# Select features
features = ["Transaction_Amount", "Transaction_Type", "Timestamp", "Location", "Device_Type", "Transaction_Status"]
X = df[features]  
y = df["Is_Fraudulent"]  

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training Data: {X_train.shape}, Testing Data: {X_test.shape}")
