import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load dataset
df = pd.read_csv("C:/fraud_detection_using_machine_learning/data/upi_fraud_data.csv")


# Step 1: Handling missing values
df.fillna({
    "Transaction_Amount": df["Transaction_Amount"].median(),
    "Transaction_Type": df["Transaction_Type"].mode()[0],
    "Timestamp": df["Timestamp"].mode()[0],
    "Location": df["Location"].mode()[0],
    "Device_Type": df["Device_Type"].mode()[0],
    "Transaction_Status": df["Transaction_Status"].mode()[0],
    "Is_Fraudulent": df["Is_Fraudulent"].mode()[0]
}, inplace=True)

# Step 2: Encoding categorical variables
label_encoders = {}
categorical_cols = ["Transaction_Type", "Location", "Device_Type", "Transaction_Status"]

for col in categorical_cols:
    label_encoders[col] = LabelEncoder()
    df[col] = label_encoders[col].fit_transform(df[col])

# Step 3: Feature Scaling (Normalize Transaction_Amount column)
scaler = StandardScaler()
df["Transaction_Amount"] = scaler.fit_transform(df[["Transaction_Amount"]])

# Save the processed dataset
df.to_csv("C:/fraud_detection_using_machine_learning/data/upi_fraud_data.csv", index=False)

# Display processed data
print("Data Preprocessing Completed!")
print(df.head())