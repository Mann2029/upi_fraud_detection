# Project Name

# AI-Based UPI Fraud Detection System (Past Transactions)

# Team Members
1. Mann Chavda (Ku2407u327)
2. Akshat Bansal (Ku2407u251)
3. Meet Dave (Ku2407u331)
4. Heman Darji (Ku2407u779)
5. Meet Vastani (Ku2407u451)

 # Overview
 - This project is an AI-based UPI fraud detection system that analyzes past transaction data (up to 300 data points) and identifies fraudulent transactions. The   system uses machine learning models to classify transactions as "Fraud" or "Not Fraud."

 ## Table of content
 1. Overview
 2. Requirements
 3. Project Structure
 4. Load Data
 5. Data Preprocessing
 6. Feature Selection
 7. Transaction Error Handling
 8. Analysing Fraud Trends
 9. Predict Fraud Transactions
 10. Model Evaluation
 11. Model Training
 12. Batch Fraud Detection
 13. Fraud Visualisation

## Requirements
Python 3.9.13

### Required Python libraries:
- pandas
- numpy
- joblib
- matplotlib
- warnings
- glob
- seaborn
- scikit-learn
- dash
- xgboost
- os

## Features
 ### Key Features of the AI-Based UPI Fraud Detection System:
1. Automated Fraud Detection – Uses machine learning models to detect fraudulent transactions.
2. Batch Fraud Detection – Analyzes multiple transactions at once to identify fraud patterns
3. Feature Selection for Better Accuracy – Uses optimized fraud-related transaction features.
4. Fraud Trend Analysis – Identifies trends in fraudulent transactions over time.
5. Visualization of Fraud Patterns – Generates various charts and graphs for fraud insights.
6. Handles Transaction Errors – Detects and corrects missing or suspicious transaction data.
7. Supports Multiple ML Models – Uses Random Forest, XGBoost, and Logistic Regression for predictions.
8. Performance Evaluation of Models – Compares models using accuracy, precision, recall, and F1-score.
9. Fraud Detection Based on Transaction Type & Amount – Analyzes fraud likelihood based on transaction behavior.
10. Fraud Location Analysis – Identifies fraud-prone regions using location-based heatmaps.

## Project Structure
AI_UPI_Fraud_Detection/
- data/
    - transactions.csv        # Main dataset (past transactions)
- models/
   - random_forest.pkl       # Saved Random Forest model
   - xgboost.pkl             # Saved XGBoost model
   - logistic_reg.pkl        # Saved Logistic Regression model
- src/
   - preprocess.py           # Data cleaning & feature engineering
   - feature_selection.py    # Selecting best fraud-related features
   - train_model.py          # Train ML models
   - evaluate_model.py       # Model performance evaluation
   - batch_fraud_detection.py # Detect fraud in multiple transactions
- visualization/
   - fraud_vs_nonfraud.png    # Fraud vs. Non-Fraud Transactions (Bar Chart).
   - fraud_vs_amount.png      # Fraud Transactions by Amount (Boxplot).
   - fraud_trend_over_time.png # Fraud Trend Over Time (Line Plot).
   - fraud_vs_amount_histogram.png # Transaction Amounts (Histogram.)
   - fraud_location_heatmap.png # Fraud by Location (Heatmap).
   - fraud_percentage_pie.png  # Fraud Percentage (Pie Chart).
   - fraud_by_transaction_type.png # Fraud by Transaction Type (Bar Chart).
  - README.md  # Project documentation

## Steps to Run the Project

1. Load Data

 - The dataset transactions.csv contains transaction details such as:
     - Transaction ID – Unique identifier for each transaction.
     - Sender ID & Receiver ID – IDs of the sender and receiver.
     - Amount – Transaction amount.
     - Location – The city or region where the transaction took place.
     - Time – Time of transaction.
     - Frequency – Number of transactions from the sender in a given period.
     - Fraud Label – Indicates whether the transaction is fraud (1) or not (0).

2️. Data Preprocessing

- Remove duplicate and null values.
- Convert categorical variables (e.g., Location) into numerical values.
- Normalize Amount and Frequency columns to ensure fair model training.

3️. Feature Selection

- Identify key fraud-related features to improve model accuracy.
- Remove irrelevant features that do not contribute to fraud detection.

4️. Transaction Error Handling

- Detect and handle missing or incorrect transaction details.
- Identify suspicious entries, such as extremely high transaction amounts.

5️. Analyzing Fraud Trends

- Generate fraud analysis graphs (see visualization/ folder).
- Understand patterns, such as fraud-prone locations, high-risk transaction times, and frequent fraud amounts.

6️. Predict Fraud Transactions

- Load trained models (xgboost.pkl, random_forest.pkl, etc.)
- Input new transaction details and predict whether it is fraudulent.

7️. Model Evaluation

- Evaluate models using performance metrics such as:

     - Accuracy – Measures correct fraud predictions.
     - Precision & Recall – Helps assess model performance in fraud detection.
     - F1-Score & ROC-AUC – Determines overall effectiveness of the model.
     - XGBoost performs best and is selected as the final model.

8️. Model Training

- Train machine learning models using past transaction data.
- Save trained models for future predictions.

9️. Batch Fraud Detection

- Use batch_fraud_detection.py to analyze multiple transactions at once and detect fraud efficiently.

10.  Fraud Visualization

- Fraud trends are visualized using:
    -  fraud_vs_nonfraud.png – Fraud vs. Non-Fraud Transactions (Bar Chart).
    -  fraud_vs_amount.png – Fraud Transactions by Amount (Boxplot).
    -  fraud_trend_over_time.png – Fraud Trend Over Time (Line Plot).
    -  fraud_vs_amount_histogram.png – Transaction Amounts for Fraud vs. Non-Fraud (Histogram).
    -  fraud_location_heatmap.png – Fraud by Location (Heatmap).
    -  fraud_percentage_pie.png – Fraud Percentage (Pie Chart).
    -  fraud_by_transaction_type.png – Fraud by Transaction Type (Bar Chart).

 ## Output Examples

- Fraud Trends Over Time – A line chart showing how fraud occurrences change over different time periods.
- Fraud by Transaction Amount – A boxplot visualizing fraudulent transactions across different amount ranges.
- Fraud by Location – A heatmap highlighting high-risk fraud locations.
- Fraud by Transaction Type – A bar chart showing which transaction types are more prone to fraud.

## Error Handling

- If no file is selected, the script exits with an error message.
- If the selected file is missing, an empty CSV file with the required columns is created.
- If required columns are missing in the dataset, the script terminates with an error.

## Future Enhancements

- Enhanced Fraud Detection – Incorporate deep learning techniques for better fraud detection accuracy.
- Additional Visualization Options – Provide more fraud analysis charts.
- Real-time Fraud Monitoring – Extend functionality to process live transactions.
- User-defined Thresholds – Allow users to set custom fraud detection thresholds.
 
  ###  Streamlit Dashboard

- A simple Streamlit web app (streamlit_app.py) has been added to visualize fraud trends in real-time.
  - To run the dashboard:
  - streamlit run streamlit_app.py
  
-The dashboard will display:
  - Fraud Trends (Line Chart)
  - Fraud Amount Distribution (Boxplot)
  - Fraud Location (Heatmap) 
  - Fraud Type Distribution (Bar Chart)

## Conclusion

- This AI-based UPI fraud detection system provides a structured approach to identifying fraudulent transactions using machine learning. By leveraging past transaction data, analyzing fraud trends, and applying predictive models, the system effectively classifies transactions as fraudulent or legitimate. Future improvements will focus on enhancing model accuracy, incorporating deep learning techniques, and enabling real-time fraud monitoring.
