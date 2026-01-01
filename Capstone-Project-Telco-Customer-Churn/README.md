# 📡 Telco Customer Churn Prediction

## 📋 Project Overview
Customer retention is critical in the telecom industry. The cost of acquiring a new customer is significantly higher than retaining an existing one. 

This project analyzes customer data to identify the key drivers of churn and builds a machine learning model to predict which customers are at high risk of leaving. The final model effectively captures **83% of churners**, allowing the marketing team to proactively target at-risk customers with retention offers.

## 💼 Business Problem
* **Goal:** Predict customer churn (Yes/No).
* **Challenge:** The dataset is highly imbalanced (only ~26% of customers churn), making standard "Accuracy" a misleading metric.
* **Value:** Identifying at-risk customers early allows for intervention (discounts, better plans), directly impacting revenue.

## 🛠️ Tech Stack
* **Language:** Python
* **Libraries:** Pandas, NumPy, Scikit-Learn, XGBoost, Imbalanced-Learn (SMOTE), Matplotlib, Seaborn.
* **Techniques:** EDA, One-Hot Encoding, SMOTE (Oversampling), Scale_Pos_Weight, Hyperparameter Tuning.

## 📊 Key Findings from EDA
1.  **Contract Type:** Customers with "Month-to-month" contracts are significantly more likely to churn compared to those with 1 or 2-year contracts.
2.  **Monthly Charges:** High monthly charges are a strong predictor of churn.
3.  **Tenure:** New customers (first 12 months) are the most volatile.

## 🤖 Model Performance
We compared three approaches to handle the imbalanced data.

| Model | Technique | Accuracy | Recall (Churn Class) | Business Impact |
| :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | Baseline | 77% | 24% | ❌ Misses most churners. |
| **Random Forest** | SMOTE (Oversampling) | 71% | 46% | ⚠️ Better, but still misses half. |
| **XGBoost** | **Weighted Loss (Champion)** | **69%** | **83%** | ✅ **Captures majority of risk.** |

*> **Note:** While Accuracy dropped slightly in the final model, the Recall for the minority class (Churners) drastically improved. In this business context, catching a Churner (Recall) is prioritized over avoiding false alarms (Precision).*

## 📈 Feature Importance (Drivers of Churn)
According to the XGBoost model, the top predictors are:
1.  **Monthly Charges**
2.  **Tenure**
3.  **Contract Type (Month-to-Month)**
4.  **Internet Service Type (Fiber Optic)**

## 🚀 How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/Plabss/ML-Journey.git
   ```
   ```bash
   cd Capstone-Project-Telco-Customer-Churn
   ```
2. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn xgboost imbalanced-learn matplotlib seaborn
   ```
3. Run the notebook:
   ```bash
   jupyter notebook "01_Data_Prep_and_EDA.ipynb"
   ```