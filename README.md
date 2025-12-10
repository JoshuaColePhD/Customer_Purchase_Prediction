# E-commerce Customer Purchase Prediction and Feature Importance Analysis

## Executive Summary

This project builds a complete end-to-end machine learning pipeline to predict whether a customer will make a purchase based on demographic and behavioral data. Using a dataset of 1,500 customers, the workflow covers:

- Data loading and exploration  
- Data preprocessing and feature engineering  
- Model training (Logistic Regression and Random Forest)  
- Model evaluation and ROC-AUC comparison  
- Insights for customer segmentation and marketing strategy  

The Random Forest classifier outperforms Logistic Regression, achieving higher accuracy, precision, recall, and a strong ROC-AUC score (~0.95).

Feature importance analysis indicates that behavioral attributes have the strongest influence on purchase likelihood:

- `TimeSpentOnWebsite` (0.187)  
- `DiscountsAvailed` (0.153)  
- `LoyaltyProgram` (0.107)

Among demographic factors, Age (0.169) and AnnualIncome (0.142) are influential predictors, while Gender (0.010) and ProductCategory (0.025) have minimal impact on purchase predictions.

Behavioral attributes play a significant role in purchase likelihood, suggesting that targeted marketing strategies could enhance conversion rates.

---

## Dataset Overview

The dataset contains **1,500 customer records** with 10 variables describing demographic attributes, behavioral patterns, and purchase outcomes.

### Data Dictionary

| Column                    | Description                                                   | Type     |
|---------------------------|---------------------------------------------------------------|----------|
| **Age**                   | Customer age (years)                                          | int64    |
| **Gender**                | Encoded gender indicator                                      | int64    |
| **AnnualIncome**          | Annual income (USD)                                           | float64  |
| **NumberOfPurchases**     | Total purchases made in the last year                         | int64    |
| **ProductCategory**       | Encoded category of products viewed or purchased              | int64    |
| **TimeSpentOnWebsite**    | Average time spent on the website (minutes)                   | float64  |
| **LoyaltyProgram**        | Loyalty program enrollment status (encoded)                   | int64    |
| **DiscountsAvailed**      | Number of discounts redeemed                                  | int64    |
| **PurchaseStatus**        | Target variable (1 = Purchased, 0 = Not Purchased)            | int64    |
| **AvgSpendingPerPurchase**| Engineered feature: AnnualIncome / (NumberOfPurchases + 1)   | float64  |

### Attribute Groups

**Demographic Attributes**
- Age  
- Gender  
- AnnualIncome  

**Behavioral Attributes**
- NumberOfPurchases  
- ProductCategory  
- TimeSpentOnWebsite  
- DiscountsAvailed  
- LoyaltyProgram  
- AvgSpendingPerPurchase  

**Target Variable**
- PurchaseStatus

---

## Data Loading and Exploration

Objectives:

- Load `customer_purchase_data.csv`  
- Inspect data types and summary statistics  
- Identify missing values  
- Visualize feature distributions  
- Explore correlations and relationships  
- Examine the distribution of the target variable  

---

## Key EDA Visuals

- Correlation heatmap (numerical features)  
- Box plots (Age, AnnualIncome, TimeSpentOnWebsite, etc. by PurchaseStatus)  
- Count plots (categorical features vs. PurchaseStatus)  
- Histograms for numerical features  

---

# Data Preprocessing and Feature Engineering

## Steps Performed

### Handling Missing Values
- Numerical features filled with median values  
- Categorical features filled with mode values  

### Feature Engineering
- Created `AvgSpendingPerPurchase` = `AnnualIncome / (NumberOfPurchases + 1)`

### Encoding Categorical Variables
- One-hot encoding for ProductCategory and Gender  
- Binary encoding for LoyaltyProgram  

### Feature Scaling
- Standardized numerical features using `StandardScaler`

---

# Model Training

## Models Used
- Logistic Regression  
- Random Forest Classifier  

## Training Process

Performed an 80/20 train-test split: