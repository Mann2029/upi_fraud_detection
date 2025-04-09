# Project Name

# Fraud detection by using machine learning.

# Team Members
1. Mann Chavda (Ku2407u327)
2. Akshat Bansal (Ku2407u251)
3. Meet Dave (Ku2407u331)
4. Heman Darji (Ku2407u779)
5. Meet Vastani (Ku2407u451)

 # Overview
 - This project is an fraud detection by using machine learning that analyzes past transaction data (up to 300 data points) and identifies fraudulent transactions. The   system uses machine learning models to classify transactions as "Fraud" or "Not Fraud."

 ## Table of content
 1. Overview
 2. Requirements
 3. Load Data
 4. Data Preprocessing
 5. Feature Selection
 6. Transaction Error Handling
 7. Analysing Fraud Trends
 8. Predict Fraud Transactions
 9. Model Evaluation
 10. Model Training
 11. Batch Fraud Detection(Visualize)
 12. Fraud Visualisation

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
 ### Key Features of Fraud Detection by using machine learrning:
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

9️. Batch Fraud Detection(

- Use batch_fraud_detection.py to analyze multiple transactions at once and detect fraud efficiently and create a predictions csv file after that csv file data will be shown into graphs and bar chart.
- Fraud_percentage_pie.png – Fraud Percentage (Pie Chart).
- Fraud_by_transaction_type.png – Fraud by Transaction Type (Bar Chart).
- Fraud_trend_over_time.png – Fraud Trend Over Time (Line Plot).
- Fraud vs. Non-Fraud Count Plot.
- Fraud by Location.

10.  Fraud Visualization

- Fraud trends are visualized using:
    - Use the Fraud_visualization.py to analyze the data of the prediction csv file.
    - There are some key visualization to show the output of he csv file:
        - Fraud cases per city.
        - Stacked Bar Chart: Fraud Cases Per City (Random Forest).
        - Fraud Predictions Across Transaction Types.
        - Model Performance Bar Chart.
    - We will use this all visualization to understand the fraud detected by the models.

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

- This project demonstrates the use of machine learning to detect fraudulent transactions based on past data. It uses models like XGBoost, Random Forest, and Logistic Regression for accurate classification. Key features include data preprocessing, fraud trend analysis, and error handling. XGBoost achieved the best performance among all models. 

