import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from PIL import Image
import matplotlib.patches as mpatches

# Define paths
RESULTS_FILE = "C:/fraud_detection_using_machine_learning/results/predictions_20250408_220842.csv"
PERFORMANCE_FILE = "C:/fraud_detection_using_machine_learning/model/model_performance.csv"
SAVE_FOLDER = "C:/fraud_detection_using_machine_learning/visualizations/"
INDIA_MAP_PATH = "C:/fraud_detection_using_machine_learning/data/map_of_india.jpg"

# Ensure save folder exists
os.makedirs(SAVE_FOLDER, exist_ok=True)

# Load dataset
df = pd.read_csv(RESULTS_FILE)

#  Check if "Fraud_Prediction_RF" exists
if "Fraud_Prediction_RF" not in df.columns:
    raise ValueError("ERROR: Column 'Fraud_Prediction_RF' is missing in the dataset!")

#  Check if "Location" exists
if "Location" not in df.columns:
    raise ValueError("ERROR: Column 'Location' is missing in the dataset!")

df["Location"] = df["Location"].str.strip().str.lower()

# Define key city coordinates
city_coords = {
    "new delhi": (28.6139, 77.2090), "mumbai": (19.0760, 72.8777), "kolkata": (22.5726, 88.3639),
    "bangalore": (12.9716, 77.5946), "hyderabad": (17.3850, 78.4867), "chennai": (13.0827, 80.2707),
    "pune": (18.5204, 73.8567), "ahmadabad": (23.0225, 72.5714), "jaipur": (26.9124, 75.7873),
    "lucknow": (26.8467, 80.9462), "bhopal": (23.2599, 77.4126), "patna": (25.5941, 85.1376),
    "indore": (22.7196, 75.8577), "nagpur": (21.1458, 79.0882), "chandigarh": (30.7333, 76.7794),
    "guwahati": (26.1445, 91.7362), "coimbatore": (11.0168, 76.9558), "bhubaneswar": (20.2961, 85.8245)
}

city_colors = plt.cm.tab20(np.linspace(0, 1, len(city_coords)))

# Load India Map
india_map = Image.open(INDIA_MAP_PATH)

# Create figure
fig, ax = plt.subplots(figsize=(10, 12))
ax.imshow(india_map, extent=[67, 98, 5, 37])

#Fraud cases per city
fraud_counts = df.groupby("Location")["Fraud_Prediction_RF"].sum()

# Debugging: Print fraud counts
print("\n🔍 Fraud Cases Per City (Random Forest):")
print(fraud_counts)

for (city, (lat, lon)), color in zip(city_coords.items(), city_colors):
    fraud_cases = fraud_counts.get(city, 0)
    fraud_size = np.log1p(fraud_cases) * 50  # Scale fraud markers
    ax.scatter(lon, lat, color=color, s=fraud_size, edgecolors="black", label=city.title(), alpha=0.75)
    ax.text(lon, lat, city.title(), fontsize=9, ha="right", color="black")

# Create legend
legend_patches = [mpatches.Patch(color=color, label=city.title()) for city, color in zip(city_coords.keys(), city_colors)]
ax.legend(handles=legend_patches, loc="lower left", fontsize=8, title="City Name", frameon=True)
ax.set_title("Fraud Cases in India (Random Forest)", fontsize=16, fontweight="bold")
ax.set_xlabel("Longitude", fontsize=12)
ax.set_ylabel("Latitude", fontsize=12)
ax.set_xlim(67, 98)
ax.set_ylim(5, 37)
ax.grid(False)  # Disable grid for better visibility
ax.axis("off")  # Hide axes for better visualization
plt.tight_layout()

plt.title("Fraud Map with City Markers and Legend")
plt.savefig(os.path.join(SAVE_FOLDER, "fraud_map_with_legend.png"))
plt.show()

# Stacked Bar Chart: Fraud Cases Per City (Random Forest)
fraud_counts_sorted = fraud_counts.sort_values(ascending=False).head(10)  # Top 10 cities

plt.figure(figsize=(12, 6))
fraud_counts_sorted.plot(kind="bar", stacked=True, color="red", alpha=0.75)
plt.title("Stacked Bar Chart - Fraud Cases Per City (Random Forest)")
plt.xlabel("City")
plt.ylabel("Number of Fraud Cases")
plt.xticks(rotation=360)
plt.ylim(0, fraud_counts_sorted.max() + 10)  # Adjust y-axis limit based on max fraud cases
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.legend(["Fraud Cases"])
plt.axhline(0, color="black", linewidth=0.8)  # Add a line at y=0 for clarity
plt.tight_layout()
plt.savefig(os.path.join(SAVE_FOLDER, "fraud_cases_rf_per_city.png"))
plt.show()

#  Fraud Predictions Across Transaction Types
# Check if 'Transaction_Type' column exists
if "Transaction_Type" in df.columns:
    # Sum fraud predictions for each transaction type
    fraud_counts = df.groupby("Transaction_Type")[["Fraud_Prediction_RF", "Fraud_Prediction_XGB", "Fraud_Prediction_LR"]].sum()

    # Plot bar chart
    fraud_counts.plot(kind="bar", figsize=(10, 6))
    plt.title("Fraud Predictions Across Transaction Types")
    plt.xlabel("Transaction Type")
    plt.ylabel("Fraudulent Transaction Count")

    # Define y-axis limits based on transaction type
    transaction_limits = {"UPI": 200, "Cash": 180, "Online": 190}
    
    # Determine max limit based on available transaction types in the dataset
    y_limit = max(transaction_limits.get(t_type, 1) for t_type in fraud_counts.index)
    plt.ylim(0, y_limit)

    # Set legend labels
    plt.legend([ "XGBoost", "Logistic Regression"])  # Added RF back for completeness
    plt.xticks(rotation=45)

    # Save the plot
    plt.savefig(os.path.join(SAVE_FOLDER, "fraud_comparison.png"))
    plt.show()
else:
    print(" WARNING: 'Transaction_Type' column missing, skipping fraud type comparison.")

#  Load Model Performance Metrics
performance_df = pd.read_csv(PERFORMANCE_FILE)

# Standardize column names
performance_df.columns = performance_df.columns.str.strip().str.replace("-", "_")

# Required columns, considering slight variations
expected_cols = {"Model", "Accuracy", "Precision", "Recall", "F1_Score"}
available_cols = set(performance_df.columns)

# Check for missing columns
missing_cols = expected_cols - available_cols
if missing_cols:
    raise ValueError(f"CSV file is missing required columns: {missing_cols}")

#  Model Performance Bar Chart
performance_df.set_index("Model")[["Accuracy", "Precision", "Recall", "F1_Score"]].plot(kind="bar", figsize=(14, 7))
plt.title("Fraud Detection Model Performance")
plt.xlabel("Model")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.legend(loc="upper right")
plt.xticks(rotation=360)
plt.tight_layout()
# Save the plot
plt.savefig(os.path.join(SAVE_FOLDER, "model_performance.png"))
plt.show()


print("\n All visualizations saved successfully!")
