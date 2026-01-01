# 🚀 Statistical Machine Learning Recap

## 📌 Overview
This repository documents my intensive journey mastering **Statistical Machine Learning**, moving from foundational statistics to advanced ensemble methods and unsupervised learning. 

Each folder contains **concept implementations**, **code experiments**, and **visualizations** built from scratch using Python. The goal of this repository is to bridge the gap between theoretical math and practical coding for industry applications.

## 🛠️ Tech Stack
* **Language:** Python 🐍
* **Libraries:** Scikit-Learn, XGBoost, NumPy, Pandas, Matplotlib, Seaborn, SciPy.
* **Environment:** Jupyter Notebooks.

---

## 📂 Repository Structure

### 1️⃣ Statistics & Foundations
*Building the mathematical intuition behind the algorithms.*
* **Data Distributions:** Visualizing Normal Distribution, Empirical Rule (68-95-99.7), and Standard Deviation.
* **Hypothesis Testing:** Implementing T-tests and understanding P-values/Statistical Significance.
* **Correlation:** Analyzing Pearson Correlation and distinguishing it from Causation.

### 2️⃣ Supervised Learning (Regression & Classification)
*Algorithms that learn from labeled data.*
* **Linear & Polynomial Regression:** Implementing the "Line of Best Fit", tackling Underfitting vs. Overfitting (Bias-Variance Tradeoff).
* **Regularization:** Using **Lasso (L1)** for feature selection and **Ridge (L2)** for handling complexity.
* **Logistic Regression:** Binary classification, Sigmoid functions, and Decision Boundaries.
* **Decision Trees & Random Forest:** Moving from single trees to Ensemble Learning (Bagging) to reduce variance.
* **Gradient Boosting (XGBoost):** Implementing Boosting (Sequential Learning) for high-performance predictions.

### 3️⃣ Unsupervised Learning
*Finding hidden patterns in unlabeled data.*
* **K-Means Clustering:** Grouping data based on Euclidean distance (and understanding its limitations).
* **DBSCAN:** Density-based clustering for identifying noise and non-spherical shapes.
* **Dimensionality Reduction:**
    * **PCA:** Linear reduction (Variance preservation).
    * **t-SNE:** Non-linear visualization of high-dimensional data (e.g., Digits dataset).

### 4️⃣ Model Optimization
*Fine-tuning for the real world.*
* **Hyperparameter Tuning:** Automating optimization using `GridSearchCV`.
* **Cross-Validation:** implementing K-Fold CV to ensure model stability.
* **Gradient Descent:** Understanding Batch vs. Stochastic vs. Mini-Batch descent.

---

## 🏆 Capstone Project: Telco Customer Churn
*> **Status:** Completed ✅*

As the final test of these skills, I built an end-to-end classification system to predict customer churn in the Telecom industry.
* **Challenge:** Highly imbalanced dataset (74% Non-Churn / 26% Churn).
* **Solution:** Compared Logistic Regression vs. Random Forest vs. XGBoost.
* **Technique:** Utilized **SMOTE** (Oversampling) and **Scale_Pos_Weight** to maximize Recall.
* **Result:** Achieved **83% Recall** on the minority class, enabling proactive customer retention.

👉 **[View the Capstone Project Here](https://github.com/Plabss/ML-Journey.git)** 

---

## 🧠 Key Learnings
1.  **Metric Selection:** Accuracy is dangerous on imbalanced data; Precision/Recall/F1 are critical.
2.  **The Trade-off:** The battle between **Bias** (Underfitting) and **Variance** (Overfitting) is central to every model choice.
3.  **Data First:** No algorithm can fix bad data. Scaling, Encoding, and Handling Outliers (DBSCAN) are prerequisites.
