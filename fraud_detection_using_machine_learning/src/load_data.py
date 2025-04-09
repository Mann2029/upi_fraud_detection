import pandas as pd

# Load dataset
df = pd.read_csv("C:/fraud_detection_using_machine_learning/data/upi_fraud_data.csv")

# Display first 5 rows
print(df.head())

# Display dataset info
print(df.info())

# Check for missing values
print(df.isnull().sum())

if df.isnull().sum().sum() == 0:
    print("No missing values in dataset")