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
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    roc_curve
)

# %% load data
df = pd.read_csv("data/customer_purchase_data.csv")

# %% inspect data
print("Dataset shape:", df.shape)
print("\nFirst five rows:\n", df.head())
print("\nData types and missing values:\n")
df.info()

print("\nStatistical summary:\n", df.describe())
print("\nMissing values count:\n", df.isnull().sum())

# Distribution of target variable
print("\nPurchase status distribution:\n", df['PurchaseStatus'].value_counts())
print("\nPurchase status proportions:\n", df['PurchaseStatus'].value_counts(normalize=True))

# %% correlation matrix
corr = df.corr(numeric_only=True)

plt.figure(figsize=(10, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Visualize target distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='PurchaseStatus', data=df)
plt.title('Purchase Status Distribution')
plt.savefig('figures/purchase_status_distribution.png')
plt.show()

# --- Data Preprocessing and Feature Engineering ---

# Define feature categories BEFORE handling missing values
categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
numerical_features.remove("PurchaseStatus")  # target should not be scaled

# Handle missing values
if df.isnull().sum().sum() > 0:
    for col in numerical_features:
        df[col].fillna(df[col].median(), inplace=True)
    for col in categorical_features:
        df[col].fillna(df[col].mode()[0], inplace=True)

# Box plots for continuous features
continuous_features = ['Age', 'AnnualIncome', 'NumberOfPurchases', 'TimeSpentOnWebsite', 'DiscountsAvailed']

plt.figure(figsize=(12, 8))
for i, col in enumerate(continuous_features, 1):
    plt.subplot(3, 2, i)
    sns.boxplot(x="PurchaseStatus", y=col, data=df)
    plt.title(f'{col} by Purchase Status')
plt.tight_layout()
plt.savefig('figures/boxplots_by_purchase_status.png')
plt.show()

# Feature engineering: avoid divide-by-zero
df['AvgSpendingPerPurchase'] = df['AnnualIncome'] / (df['NumberOfPurchases'] + 1)
print(df['AvgSpendingPerPurchase'].head())


# --- Model Training ---

# Prepare data
X = df.drop('PurchaseStatus', axis=1)
y = df['PurchaseStatus']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Categorical features:", categorical_features)
print("Numerical features:", numerical_features)

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

# Logistic Regression Pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42))
])

# Fit model
model.fit(X_train, y_train)

# Evaluate Logistic Regression
y_pred = model.predict(X_test)
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm)
disp.plot()
plt.savefig('figures/confusion_matrix_logistic.png')
plt.show()


# --- Advanced Model Training and Evaluation ---

advanced_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

advanced_model.fit(X_train, y_train)

y_pred_advanced = advanced_model.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_advanced))
print("\nClassification Report:\n", classification_report(y_test, y_pred_advanced))

cm = confusion_matrix(y_test, y_pred_advanced)
disp = ConfusionMatrixDisplay(cm)
disp.plot()
plt.savefig('figures/confusion_matrix_random_forest.png')
plt.show()

# ROC-AUC for Logistic Regression
y_prob = model.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_prob)
fpr, tpr, _ = roc_curve(y_test, y_prob)

print("ROC-AUC (Logistic Regression):", roc_auc)

plt.figure()
plt.plot(fpr, tpr, label=f'Logistic Regression (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.savefig('figures/roc_curve_logistic_regression.png')
plt.show()

# ROC-AUC for Random Forest
y_prob_adv = advanced_model.predict_proba(X_test)[:, 1]
roc_auc_adv = roc_auc_score(y_test, y_prob_adv)
fpr_adv, tpr_adv, _ = roc_curve(y_test, y_prob_adv)

print("ROC-AUC (Random Forest):", roc_auc_adv)

plt.figure()
plt.plot(fpr_adv, tpr_adv, label=f'Random Forest (AUC = {roc_auc_adv:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.savefig('figures/roc_curve_random_forest.png')
plt.show()

# Combined ROC Curve
plt.figure()
plt.plot(fpr, tpr, label=f'Logistic Regression (AUC = {roc_auc:.2f})')
plt.plot(fpr_adv, tpr_adv, label=f'Random Forest (AUC = {roc_auc_adv:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.title("ROC Curve Comparison")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.savefig('figures/roc_curve_comparison.png')
plt.show()