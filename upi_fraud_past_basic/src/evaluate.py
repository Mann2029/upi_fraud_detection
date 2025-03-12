import pandas as pd
import os

#  Define file paths
predictions_path = "C:/upi_fraud_past_basic/results/predictions.csv"
actual_data_path = "C:/upi_fraud_past_basic/data/new_transactions.csv"

# Check if required files exist
if not os.path.exists(predictions_path):
    print(f" ERROR: Predictions file not found at {predictions_path}")
    exit()

if not os.path.exists(actual_data_path):
    print(f" ERROR: Actual transactions file not found at {actual_data_path}")
    exit()

#  Load predictions and actual data
try:
    predictions_df = pd.read_csv(predictions_path)
    actual_df = pd.read_csv(actual_data_path)
    print("Files loaded successfully!")
except Exception as e:
    print(f" ERROR: Failed to load CSV files. {e}")
    exit()

# Ensure required columns exist
required_columns = ["Transaction_ID", "Is_Fraudulent"]
if not set(required_columns).issubset(actual_df.columns):
    missing_cols = list(set(required_columns) - set(actual_df.columns))
    print(f"ERROR: Missing columns in actual transactions file -> {missing_cols}")
    exit()

if "Is_Fraudulent_Predicted" not in predictions_df.columns:
    print(f" ERROR: Missing 'Is_Fraudulent_Predicted' column in predictions.csv")
    exit()

# Merge actual data and predictions
merged_df = actual_df.merge(predictions_df, on="Transaction_ID", how="inner")

# Check if the merge was successful
if merged_df.empty:
    print(" ERROR: No matching transactions found between actual data and predictions.")
    exit()

#  Compare predictions with actual labels
merged_df["Correct_Prediction"] = merged_df["Is_Fraudulent"] == merged_df["Is_Fraudulent_Predicted"]

#  Compute evaluation metrics
accuracy = merged_df["Correct_Prediction"].mean()
total_transactions = len(merged_df)
correct_predictions = merged_df["Correct_Prediction"].sum()
incorrect_predictions = total_transactions - correct_predictions

#  Display evaluation results
print("\n📊 EVALUATION RESULTS 📊")
print(f"✅ Total Transactions Evaluated: {total_transactions}")
print(f"✅ Correct Predictions: {correct_predictions}")
print(f"❌ Incorrect Predictions: {incorrect_predictions}")
print(f"🎯 Accuracy: {accuracy:.2%}")

#  Save evaluation results
output_dir = "C:/upi_fraud_past_basic/results/"
os.makedirs(output_dir, exist_ok=True)  # Ensure directory exists
evaluation_path = os.path.join(output_dir, "evaluation_results.csv")

merged_df.to_csv(evaluation_path, index=False)
print(f"Evaluation results saved successfully to {evaluation_path}!")
