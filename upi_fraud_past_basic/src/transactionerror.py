import pandas as pd

# Load the transactions file
file_path = "C:/upi_fraud_past_basic/data/new_transactions.csv"
df = pd.read_csv(file_path)

# Add a dummy 'Is_Fraudulent' column (Replace this with real labels)
df["Is_Fraudulent"] = 0  # Default: No fraud (Modify actual fraud labels if available)

# Save the corrected file
df.to_csv(file_path, index=False)

print("'Is_Fraudulent' column added successfully!")

prediction_path = "C:/upi_fraud_past_basic/results/predictions.csv"
df_pred = pd.read_csv(prediction_path)
print(df_pred.columns)

