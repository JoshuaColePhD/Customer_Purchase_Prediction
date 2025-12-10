# --- Data Loading and Exploration ---

# %% import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
)

# %% load data
# Load the e-commerce customer dataset
df = pd.read_csv("data/customer_purchase_data.csv")

# %% inspect data
# Display basic information about the dataset 
print("Dataset shape:", df.shape)
print("\nFirst five rows:")
print(df.head())
print("\nData types and missing values:")

# Display data types and missing values
df.info()
print("\nStatistical summary:")
print(df.describe())
print("\nMissing values count:")
print(df.isnull().sum())

# Distribution of target variable
print("\nPurchase status distribution:")
print(df['PurchaseStatus'].value_counts())
print("\nPurchase status proportions:")
print(df['PurchaseStatus'].value_counts(normalize=True))

# %% visualize data
# Compute correlation matrix for numerical features
corr = df.corr(numeric_only=True)

plt.figure(figsize=(10, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix') 
plt.show()

# overall correlations are weak, but this is expected for a primarily categorical problem

# Visualize target distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='PurchaseStatus', data=df)
plt.title('Purchase Status Distribution')
plt.savefig('figures/purchase_status_distribution.png')
plt.show()

# --- Data Preprocessing and Feature Engineering ---

# Define feature types
categorical_features = ['Gender', 'ProductCategory', 'LoyaltyProgram']
numerical_features = [
    'Age',
    'AnnualIncome',
    'NumberOfPurchases',
    'TimeSpentOnWebsite',
    'DiscountsAvailed',
]

# Handle missing values if any
if df.isnull().sum().sum() > 0:
    # Simple imputation: median for numerical, mode for categorical
    for col in numerical_features:
        df[col].fillna(df[col].median(), inplace=True)
    for col in categorical_features:
        df[col].fillna(df[col].mode()[0], inplace=True)

# Box plots for continuous features by purchase status
continuous_features = [
    'Age',
    'AnnualIncome',
    'NumberOfPurchases',
    'TimeSpentOnWebsite',
    'DiscountsAvailed'
]

plt.figure(figsize=(12, 8))
for i, col in enumerate(continuous_features, 1):
    plt.subplot(3, 2, i)
    sns.boxplot(x="PurchaseStatus", y=col, data=df)
    plt.title(f'{col} by Purchase Status')
plt.tight_layout()
plt.savefig('figures/boxplots_by_purchase_status.png')
plt.show()

# Feature engineering
# Create a feature for average spending per purchase
df['AvgSpendingPerPurchase'] = df['AnnualIncome'] / (df['NumberOfPurchases'].replace(0, 1))
print(df['AvgSpendingPerPurchase'].head())

# --- Model Training and Baseline Evaluation ---

# %% Feature and Target 
# Separate features and target variable
X = df.drop('PurchaseStatus', axis=1)
y = df['PurchaseStatus']

# %% Train/test split
# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# %% Identify feature types
# Identify categorical and numerical features
categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

print("Categorical features:", categorical_features)
print("Numerical features:", numerical_features)

# %% Preprocessing pipeline
# ColumnTransformer applies different preprocessing to different feature types
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

# Pipeline: preprocessing + model (Logistic Regression)
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42))
])

# %% Fit baseline model
# Fit the full pipeline on the training data
model.fit(X_train, y_train)

# %% Evaluate on test data
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report: \n")
print(classification_report(y_test, y_pred))

# visualize confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.savefig('figures/confusion_matrix.png')
plt.show()

# --- Advanced Model Training and Evaluation ---
# %% Advanced model: Random Forest
# Use the same preprocessing but with Random Forest Classifier
advanced_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# %% Fit advanced model
advanced_model.fit(X_train, y_train)

# %% Evaluate advanced model
y_pred_advanced = advanced_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred_advanced))
print("\nClassification Report: \n")
print(classification_report(y_test, y_pred_advanced))

# visualize confusion matrix
cm = confusion_matrix(y_test, y_pred_advanced)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.savefig('figures/confusion_matrix_random_forest.png')
plt.show()

# --- Feature Importance (Random Forest) ---

# %% 
# --- Feature Importance (Random Forest) ---

# Get fitted preprocessor and RF model from the pipeline
preprocessor = advanced_model.named_steps['preprocessor']
rf_model = advanced_model.named_steps['classifier']

# Get feature names from the preprocessor
# This works in recent sklearn versions
feature_names = preprocessor.get_feature_names_out()

# Get feature importances from the RF
importances = rf_model.feature_importances_

# Build DataFrame
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print("\nRandom Forest Feature Importances:\n")
print(feature_importance_df)

# Plot top features
plt.figure(figsize=(10, 6))
sns.barplot(x="Importance", y="Feature", data=feature_importance_df.head(15))
plt.title('Top 15 Feature Importances (Random Forest)')
plt.tight_layout()
plt.savefig('figures/feature_importance_random_forest.png')
plt.show()

# %% ROC-AUC for Logistic Regression
# Get predicted probabilities for the positive class
y_prob = model.predict_proba(X_test)[:, 1]

# Calculate ROC-AUC for Logistic Regression
roc_auc = roc_auc_score(y_test, y_prob)
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

print("ROC-AUC Score (Logistic Regression):", roc_auc)

plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.savefig('figures/roc_curve_logistic_regression.png')
plt.show()

# %% ROC-AUC for Random Forest
# Get predicted probabilities for the positive class from Random Forest
y_prob_advanced = advanced_model.predict_proba(X_test)[:, 1]

# Calculate ROC-AUC for Random Forest
roc_auc_advanced = roc_auc_score(y_test, y_prob_advanced)
fpr_adv, tpr_adv, thresholds_adv = roc_curve(y_test, y_prob_advanced)

print("ROC-AUC Score (Random Forest):", roc_auc_advanced)

plt.figure()
plt.plot(fpr_adv, tpr_adv, label='Random Forest (area = {:.2f})'.format(roc_auc_advanced))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.savefig('figures/roc_curve_random_forest.png')
plt.show()

# %% Combined ROC Curve for Logistic Regression and Random Forest
# Plot both ROC curves on the same graph for comparison
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = {:.2f})'.format(roc_auc))
plt.plot(fpr_adv, tpr_adv, label='Random Forest (area = {:.2f})'.format(roc_auc_advanced))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend(loc='lower right')
plt.savefig('figures/roc_curve_comparison.png')
plt.show()
