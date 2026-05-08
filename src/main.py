"""Customer purchase prediction pipeline.

This script trains interpretable baseline and nonlinear classification models
for a marketing use case: ranking customers by purchase propensity so campaign
teams can target the highest-likelihood conversion opportunities.
"""

import os
from pathlib import Path

import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "customer_purchase_data.csv"
FIGURES_DIR = PROJECT_ROOT / "figures"
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib_cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

TARGET = "PurchaseStatus"

NUMERIC_FEATURES = [
    "Age",
    "AnnualIncome",
    "NumberOfPurchases",
    "TimeSpentOnWebsite",
    "DiscountsAvailed",
    "AvgSpendingPerPurchase",
]
CATEGORICAL_FEATURES = ["Gender", "ProductCategory", "LoyaltyProgram"]


class PurchaseFeatureEngineer(BaseEstimator, TransformerMixin):
    """Create behavior features before model-specific preprocessing."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # Captures customer economic value normalized by purchase frequency.
        # The +1 keeps zero-purchase customers in the sample without exploding
        # the feature, which matters when predicting first or next conversion.
        X["AvgSpendingPerPurchase"] = X["AnnualIncome"] / (
            X["NumberOfPurchases"] + 1
        )
        return X


def load_data(path=DATA_PATH):
    """Load source data and keep target validation close to ingestion."""
    df = pd.read_csv(path)
    if TARGET not in df.columns:
        raise ValueError(f"Expected target column '{TARGET}' in {path}")
    return df


def make_preprocessor():
    """Build leakage-safe preprocessing for numeric and categorical signals."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def make_models():
    """Return candidate models with identical preprocessing contracts."""
    return {
        "Logistic Regression": Pipeline(
            steps=[
                ("features", PurchaseFeatureEngineer()),
                ("preprocessor", make_preprocessor()),
                (
                    "classifier",
                    LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("features", PurchaseFeatureEngineer()),
                ("preprocessor", make_preprocessor()),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=300,
                        min_samples_leaf=5,
                        random_state=RANDOM_STATE,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
    }


def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    """Fit one model and return portfolio-ready performance metrics."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # Cross-validated ROC-AUC is the stability check; the holdout set remains
    # the clean final estimate used for business-facing performance claims.
    cv_auc = cross_val_score(
        model, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=1
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_score),
        "cv_roc_auc_mean": cv_auc.mean(),
        "cv_roc_auc_std": cv_auc.std(),
    }
    return metrics, y_pred, y_score


def plot_target_distribution(df):
    plt.figure(figsize=(7, 4))
    ax = sns.countplot(
        x=TARGET,
        hue=TARGET,
        data=df,
        palette=["#6c757d", "#2a9d8f"],
        legend=False,
    )
    ax.set_title("Customer Purchase Outcome Mix")
    ax.set_xlabel("Purchase outcome")
    ax.set_ylabel("Number of customers")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["No purchase", "Purchased"])
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "purchase_status_distribution.png", dpi=160)
    plt.close()


def plot_behavior_boxplots(df):
    engineered = PurchaseFeatureEngineer().transform(df)
    continuous_features = [
        "Age",
        "AnnualIncome",
        "NumberOfPurchases",
        "TimeSpentOnWebsite",
        "DiscountsAvailed",
        "AvgSpendingPerPurchase",
    ]

    plt.figure(figsize=(13, 8))
    for i, col in enumerate(continuous_features, 1):
        plt.subplot(2, 3, i)
        sns.boxplot(
            x=TARGET,
            y=col,
            hue=TARGET,
            data=engineered,
            palette=["#6c757d", "#2a9d8f"],
            legend=False,
        )
        plt.title(f"{col} by Purchase Outcome")
        plt.xlabel("")
        plt.xticks([0, 1], ["No purchase", "Purchased"])
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "boxplots_by_purchase_status.png", dpi=160)
    plt.close()


def plot_confusion_matrix(name, y_test, y_pred, filename):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["No purchase", "Purchased"]
    )
    disp.plot(cmap="Blues", values_format="d")
    plt.title(f"{name}: Campaign Targeting Outcomes")
    plt.xlabel("Predicted customer outcome")
    plt.ylabel("Actual customer outcome")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=160)
    plt.close()


def plot_roc_curves(results, y_test):
    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    for name, payload in results.items():
        RocCurveDisplay.from_predictions(
            y_test,
            payload["y_score"],
            name=f"{name} (AUC = {payload['metrics']['roc_auc']:.3f})",
            ax=ax,
        )
    ax.plot([0, 1], [0, 1], linestyle="--", color="#6c757d", label="No-skill model")
    ax.set_title("Model Ability to Rank Likely Purchasers")
    ax.set_xlabel("False positive rate: outreach wasted on non-buyers")
    ax.set_ylabel("True positive rate: purchasers captured")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "roc_curve_comparison.png", dpi=160)
    plt.close()

    for name, payload in results.items():
        slug = name.lower().replace(" ", "_")
        plt.figure(figsize=(7, 5))
        ax = plt.gca()
        RocCurveDisplay.from_predictions(
            y_test,
            payload["y_score"],
            name=f"{name} (AUC = {payload['metrics']['roc_auc']:.3f})",
            ax=ax,
        )
        ax.plot([0, 1], [0, 1], linestyle="--", color="#6c757d")
        ax.set_title(f"{name}: Purchase Propensity Ranking")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"roc_curve_{slug}.png", dpi=160)
        plt.close()


def get_random_forest_importance(model):
    """Aggregate fitted importances into business-readable feature names."""
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]
    feature_names = preprocessor.get_feature_names_out()

    importances = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": classifier.feature_importances_,
        }
    )
    importances["feature"] = (
        importances["feature"]
        .str.replace("num__", "", regex=False)
        .str.replace("cat__", "", regex=False)
        .str.replace(r"_(\d+)$", "", regex=True)
    )

    return (
        importances.groupby("feature", as_index=False)["importance"]
        .sum()
        .sort_values("importance", ascending=False)
    )


def plot_feature_importance(feature_importance):
    top_features = feature_importance.head(10).sort_values("importance")
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(
        x="importance",
        y="feature",
        data=top_features,
        color="#2a9d8f",
    )
    ax.set_title("Top Purchase Drivers for Campaign Prioritization")
    ax.set_xlabel("Relative influence on purchase propensity")
    ax.set_ylabel("")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "feature_importance_random_forest.png", dpi=160)
    plt.close()


def estimate_business_impact(
    y_test, y_score, contact_rate=0.20, revenue_per_purchase=100
):
    """Translate ranking quality into a simple campaign sizing scenario."""
    scored = pd.DataFrame({"actual": y_test.to_numpy(), "score": y_score})
    top_segment = scored.nlargest(int(len(scored) * contact_rate), "score")
    baseline_rate = scored["actual"].mean()
    targeted_rate = top_segment["actual"].mean()
    incremental_purchases = (targeted_rate - baseline_rate) * len(top_segment)

    return {
        "contact_rate": contact_rate,
        "baseline_conversion_rate": baseline_rate,
        "targeted_conversion_rate": targeted_rate,
        "conversion_lift": targeted_rate / baseline_rate,
        "incremental_purchases": incremental_purchases,
        "estimated_incremental_revenue": incremental_purchases * revenue_per_purchase,
    }


def main():
    sns.set_theme(style="whitegrid")
    FIGURES_DIR.mkdir(exist_ok=True)
    Path(os.environ["MPLCONFIGDIR"]).mkdir(exist_ok=True)

    df = load_data()
    X = df.drop(columns=TARGET)
    y = df[TARGET]

    print(f"Dataset shape: {df.shape}")
    print("\nPurchase status proportions:")
    print(y.value_counts(normalize=True).rename("share"))

    plot_target_distribution(df)
    plot_behavior_boxplots(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    results = {}
    for name, model in make_models().items():
        metrics, y_pred, y_score = evaluate_model(
            name, model, X_train, X_test, y_train, y_test
        )
        results[name] = {
            "model": model,
            "metrics": metrics,
            "y_pred": y_pred,
            "y_score": y_score,
        }

        print(f"\n{name}")
        metric_summary = pd.Series({k: v for k, v in metrics.items() if k != "model"})
        print(metric_summary.round(4))
        print(
            classification_report(
                y_test, y_pred, target_names=["No purchase", "Purchased"]
            )
        )

    plot_confusion_matrix(
        "Logistic Regression",
        y_test,
        results["Logistic Regression"]["y_pred"],
        "confusion_matrix_logistic.png",
    )
    plot_confusion_matrix(
        "Random Forest",
        y_test,
        results["Random Forest"]["y_pred"],
        "confusion_matrix_random_forest.png",
    )
    plot_roc_curves(results, y_test)

    feature_importance = get_random_forest_importance(results["Random Forest"]["model"])
    feature_importance.to_csv(
        FIGURES_DIR / "feature_importance_random_forest.csv", index=False
    )
    plot_feature_importance(feature_importance)

    impact = estimate_business_impact(
        y_test, results["Random Forest"]["y_score"], contact_rate=0.20
    )

    print("\nModel comparison:")
    print(pd.DataFrame([payload["metrics"] for payload in results.values()]).round(4))

    print("\nRandom Forest feature importance:")
    print(feature_importance.round(4).to_string(index=False))

    print("\nBusiness impact scenario: target top 20% by predicted purchase propensity")
    print(pd.Series(impact).round(4))


if __name__ == "__main__":
    main()
