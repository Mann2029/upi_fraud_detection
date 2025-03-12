import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Define file paths
RESULTS_FOLDER = "C:/upi_fraud_past_basic/results/"
PREDICTIONS_FILE = os.path.join(RESULTS_FOLDER, "predictions.csv")

# Load data
if not os.path.exists(PREDICTIONS_FILE):
    print(" ERROR: Predictions file not found!")
    exit()

df = pd.read_csv(PREDICTIONS_FILE)

#  Ensure required columns exist
if "Is_Fraudulent_Predicted" not in df.columns:
    print(" ERROR: Missing 'Is_Fraudulent_Predicted' column in predictions.csv")
    exit()

#  **1. Fraud vs. Non-Fraud Count Plot**
plt.figure(figsize=(6, 4))
sns.countplot(x=df["Is_Fraudulent_Predicted"], hue=df["Is_Fraudulent_Predicted"], palette={0: "blue", 1: "red"}, legend=False)
plt.title("Fraud vs. Non-Fraud Transactions")
plt.xlabel("Fraudulent (1) vs. Non-Fraudulent (0)")
plt.ylabel("Count")
plt.savefig(os.path.join(RESULTS_FOLDER, "fraud_vs_non_fraud.png"))
plt.close()
print(" Fraud vs. Non-Fraud Transactions saved!")

#  **2. Fraud by Transaction Amount (Boxplot)**
plt.figure(figsize=(6, 4))
sns.boxplot(
    x=df["Is_Fraudulent_Predicted"].astype(str),  
    y=df["Amount"],  
    hue=df["Is_Fraudulent_Predicted"].astype(str),  # Explicit hue
    palette={"0": "green", "1": "red"},
    legend=False  
)
plt.title("Fraud vs. Transaction Amount")
plt.xlabel("Fraudulent (1) vs. Non-Fraudulent (0)")
plt.ylabel("Transaction Amount")
plt.savefig(os.path.join(RESULTS_FOLDER, "fraud_vs_transaction_amount.png"))
plt.close()

print(" Fraud vs. Transaction Amount boxplot saved!")

#  **3. Fraud Trend Over Time (Line Chart)**
if "Timestamp" in df.columns:
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    fraud_over_time = df[df["Is_Fraudulent_Predicted"] == 1].groupby(df["Timestamp"].dt.date).size()

    plt.figure(figsize=(8, 5))
    fraud_over_time.plot(kind="line", color="red", marker="o")
    plt.title("Fraud Trend Over Time")
    plt.xlabel("Date")
    plt.ylabel("Number of Fraud Cases")
    plt.grid(True)
    plt.savefig(os.path.join(RESULTS_FOLDER, "fraud_trend_over_time.png"))
    plt.close()
    print(" Fraud trend over time saved!")

#  **4. Fraud by Location (Top 10 Cities)**
if "Location" in df.columns:
    fraud_by_location = df[df["Is_Fraudulent_Predicted"] == 1]["Location"].value_counts().head(10)

    plt.figure(figsize=(8, 5))
    sns.barplot(x=fraud_by_location.values, y=fraud_by_location.index, hue=fraud_by_location.index, dodge=False, palette="Reds")
    plt.title("Top 10 Locations with Most Fraudulent Transactions")
    plt.xlabel("Number of Fraud Cases")
    plt.ylabel("Location")
    plt.legend([],[], frameon=False)  # Hide legend
    plt.savefig(os.path.join(RESULTS_FOLDER, "fraud_by_location.png"))
    plt.close()
    print(" Fraud by location saved!")

#  **5. Fraud vs. Non-Fraud Transaction Amount (Histogram)**
plt.figure(figsize=(8, 5))
sns.histplot(df[df["Is_Fraudulent_Predicted"] == 1]["Amount"], bins=30, color="red", label="Fraud", kde=True)
sns.histplot(df[df["Is_Fraudulent_Predicted"] == 0]["Amount"], bins=30, color="blue", label="Non-Fraud", kde=True)
plt.title("Transaction Amount Distribution (Fraud vs. Non-Fraud)")
plt.xlabel("Transaction Amount")
plt.ylabel("Frequency")
plt.legend()
plt.savefig(os.path.join(RESULTS_FOLDER, "fraud_vs_transaction_histogram.png"))
plt.close()
print(" Fraud vs. Non-Fraud transaction amount histogram saved!")

#  **6. Fraud Percentage Pie Chart**
fraud_count = df["Is_Fraudulent_Predicted"].value_counts()
labels = ["Non-Fraud", "Fraud"]
colors = ["blue", "red"]

plt.figure(figsize=(6, 6))
plt.pie(fraud_count, labels=labels, autopct="%1.1f%%", colors=colors, startangle=140, explode=[0, 0.1], shadow=True)
plt.title("Fraud Percentage in Transactions")
plt.savefig(os.path.join(RESULTS_FOLDER, "fraud_percentage_pie_chart.png"))
plt.close()
print(" Fraud percentage pie chart saved!")

#  **7. Fraud by Transaction Type**
if "Transaction_Type" in df.columns:
    plt.figure(figsize=(8, 5))
    sns.countplot(y=df[df["Is_Fraudulent_Predicted"] == 1]["Transaction_Type"], hue=df[df["Is_Fraudulent_Predicted"] == 1]["Transaction_Type"], dodge=False, palette="Reds")
    plt.title("Fraud by Transaction Type")
    plt.xlabel("Count")
    plt.ylabel("Transaction Type")
    plt.legend([],[], frameon=False)  # Hide legend
    plt.savefig(os.path.join(RESULTS_FOLDER, "fraud_by_transaction_type.png"))
    plt.close()
    print(" Fraud by transaction type saved!")

print("📊 All 7 fraud visualizations saved in the results folder!")