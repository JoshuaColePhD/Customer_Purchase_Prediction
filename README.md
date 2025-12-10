# 📘 E-commerce Customer Purchase Prediction and Feature Importance Analysis

![Python](https://img.shields.io/badge/Python-3.13-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-orange)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-green)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

---

# 📂 Project Overview

This project builds a complete end-to-end **machine learning pipeline** to predict whether a customer will make a purchase based on demographic and behavioral data. The workflow includes:  
data loading, exploration, preprocessing, feature engineering, model training, evaluation, and insight generation.

---

# 🧾 Executive Summary

Using a dataset of **1,500 customers**, this project compares two classification models—**Logistic Regression** and **Random Forest**—to predict purchase behavior.

Key steps include:

- Exploratory data analysis  
- Data cleaning and preprocessing  
- Feature engineering  
- Training and evaluating two classification models  
- Interpreting feature importance  
- Generating actionable business insights  

### 🔑 Key Findings

- **Random Forest outperforms Logistic Regression** with higher accuracy, precision, recall, and ROC-AUC.  
- Behavioral attributes are the strongest predictors of purchase likelihood.  
- The most influential features include:  
  - `TimeSpentOnWebsite` (0.187)  
  - `DiscountsAvailed` (0.153)  
  - `LoyaltyProgram` (0.107)  
- Demographic features play a secondary role, with **Gender** and **ProductCategory** contributing minimally.

These results suggest that **behavior-driven marketing strategies** are more effective than demographic segmentation.

## 📈 Model Performance Visualization

Below is the combined ROC curve comparing Logistic Regression and Random Forest performance on the test set:

![ROC Curve Comparison](figures/roc_curve_comparison.png)

---

# 📊 Dataset Overview

The dataset contains **1,500 rows** and **10 variables** capturing demographic characteristics, web behavior, and purchase outcomes.

### Data Dictionary

| Column                    | Description                                                   | Type     |
|---------------------------|---------------------------------------------------------------|----------|
| Age                       | Customer age (years)                                          | int64    |
| Gender                    | Encoded gender indicator                                      | int64    |
| AnnualIncome              | Annual income (USD)                                           | float64  |
| NumberOfPurchases         | Total purchases in the last year                              | int64    |
| ProductCategory           | Encoded product category viewed or purchased                  | int64    |
| TimeSpentOnWebsite        | Average time spent on the website (minutes)                   | float64  |
| LoyaltyProgram            | Whether user is enrolled in loyalty program                   | int64    |
| DiscountsAvailed          | Number of discounts redeemed                                  | int64    |
| PurchaseStatus            | Target variable (1 = Purchased, 0 = Not Purchased)            | int64    |
| AvgSpendingPerPurchase    | Engineered: AnnualIncome / (NumberOfPurchases + 1)            | float64  |

### Attribute Groups

- **Demographic:** Age, Gender, AnnualIncome  
- **Behavioral:** NumberOfPurchases, ProductCategory, TimeSpentOnWebsite, DiscountsAvailed, LoyaltyProgram, AvgSpendingPerPurchase  
- **Target:** PurchaseStatus  

---

# 🔍 Data Loading and Exploration

**EDA Objectives**

- Review data types and summary stats  
- Check for missing values  
- Explore distributions and correlations  
- Assess target class balance  
- Visualize demographic and behavioral patterns  

**Key Visuals**

- Correlation heatmap  
- Box plots comparing features by purchase status  
- Count plots of categorical variables  
- Histograms for numerical features  

---

# 🛠️ Preprocessing and Feature Engineering

### Steps Performed

**Missing Values**  
- Numerical: filled using median  
- Categorical: filled using mode  

**Feature Engineering**  
- Created `AvgSpendingPerPurchase` to capture spending behavior.

**Encoding**  
- One-hot encoding for `ProductCategory` and `Gender`  
- Binary encoding for `LoyaltyProgram`  

**Scaling**  
- StandardScaler applied to numerical features  

Preprocessing was implemented in a **ColumnTransformer** for consistency across models.

---

# 🤖 Model Training

Two classification models were trained:

- **Logistic Regression** (baseline)  
- **Random Forest Classifier** (nonlinear and ensemble-based)  

A standard **80/20 train-test split** was used.

---

# 🚀 Advanced Model Evaluation

### 🔧 Model Performance

| Model                 | Accuracy | ROC–AUC | Key Observations |
|----------------------|----------|---------|------------------|
| Logistic Regression  | **0.8367** | **0.8989** | Interpretable baseline; lower recall for purchasers |
| Random Forest        | **0.9600** | **0.9544** | Best performer; fewer misclassifications; strong class separation |

### 📊 Classification Insights

- Logistic Regression struggles with nonlinear behavioral patterns.  
- Random Forest substantially improves precision, recall, and AUC, making it ideal for customer purchase prediction.  

### 📈 Visualization Outputs (saved in `figures/`)

- confusion_matrix_random_forest.png  
- roc_curve_logistic_regression.png  
- roc_curve_random_forest.png  
- roc_curve_comparison.png  

These plots highlight the superior predictive performance of Random Forest.

---

# 🌟 Feature Importance Analysis (Random Forest)

| Feature                     | Importance |
|----------------------------|-----------:|
| TimeSpentOnWebsite         | 0.1873 |
| Age                        | 0.1686 |
| DiscountsAvailed           | 0.1534 |
| AnnualIncome               | 0.1419 |
| LoyaltyProgram             | 0.1066 |
| NumberOfPurchases          | 0.1034 |
| AvgSpendingPerPurchase     | 0.1032 |
| ProductCategory            | 0.0254 |
| Gender                     | 0.0102 |

### 🔍 Insight  
Behavioral factors dominate predictive power — particularly **engagement, discount behavior, and loyalty participation**.

---

# 💼 Business Recommendations

1. **Target high-engagement users**  
   Customers spending more time on-site show high purchase intent.

2. **Strengthen loyalty program offerings**  
   LoyaltyProgram is a strong predictor; enhanced incentives may drive conversions.

3. **Prioritize behavioral segmentation**  
   Behavioral features outperform demographic ones in predicting purchases.

4. **Optimize discount-driven promotions**  
   Discount users are more likely to convert; personalized offers could increase revenue.

---