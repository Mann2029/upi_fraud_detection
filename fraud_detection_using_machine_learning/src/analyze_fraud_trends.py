import pandas as pd
import os
import glob

#  Define results path
RESULTS_FOLDER = "C:/fraud_detection_using_machine_learning/results/"
REPORT_FILE = os.path.join(RESULTS_FOLDER, "fraud_trends_report.csv")

#  Find all prediction files
prediction_files = glob.glob(os.path.join(RESULTS_FOLDER, "predictions_*.csv"))

if not prediction_files:
    print(" ERROR: No prediction files found!")
    exit()

#  Load all batch prediction files
all_data = []
for file in prediction_files:
    try:
        df = pd.read_csv(file)
        df["Batch_File"] = os.path.basename(file)  # Track which file data came from
        all_data.append(df)
    except Exception as e:
        print(f" ERROR loading {file}: {e}")

# Combine all batches into a single DataFrame
fraud_data = pd.concat(all_data, ignore_index=True)

# Ensure required columns exist
required_columns = ["Fraud_Prediction_RF", "Fraud_Prediction_XGB", "Fraud_Prediction_LR"]
missing_cols = [col for col in required_columns if col not in fraud_data.columns]
if missing_cols:
    print(f" ERROR: Missing required columns -> {missing_cols}")
    exit()

# Compute fraud detection insights
fraud_data["Is_Fraudulent"] = fraud_data[["Fraud_Prediction_RF", "Fraud_Prediction_XGB", "Fraud_Prediction_LR"]].mean(axis=1) >= 0.5

total_transactions = len(fraud_data)
total_fraud_cases = fraud_data["Is_Fraudulent"].sum()
fraud_rate = (total_fraud_cases / total_transactions) * 100

#  Analyze fraud trends by time (if timestamp column exists)
if "Timestamp" in fraud_data.columns:
    fraud_data["Timestamp"] = pd.to_datetime(fraud_data["Timestamp"])
    fraud_trends = fraud_data.resample("D", on="Timestamp")["Is_Fraudulent"].mean() * 100
else:
    fraud_trends = None

#  Save fraud summary report
summary = pd.DataFrame({
    "Total_Transactions": [total_transactions],
    "Total_Fraud_Cases": [total_fraud_cases],
    "Fraud_Rate (%)": [fraud_rate]
})
summary.to_csv(REPORT_FILE, index=False)

print("\n📊 FRAUD TRENDS REPORT 📊")
print(summary)
print(f" Fraud trends report saved: {REPORT_FILE}")

if fraud_trends is not None:
    print("\n📈 Fraud Trends Over Time (Daily Fraud Rate %):")
    print(fraud_trends.head())