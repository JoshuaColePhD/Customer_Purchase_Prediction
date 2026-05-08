# Customer Purchase Prediction

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4%2B-orange)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

## Business Problem

Marketing teams often have more prospective customers than budgeted outreach capacity. This project builds a purchase propensity model that ranks customers by likelihood to buy, allowing a business to prioritize high-intent customers, reduce wasted campaign spend, and tune the trade-off between reach and conversion quality.

The modeling question is:

> Which customers are most likely to purchase, and which behavioral signals should drive campaign targeting?

## Executive Summary

Using 1,500 customer records with demographic, purchase-history, website-engagement, loyalty, and discount features, I compared an interpretable Logistic Regression baseline against a nonlinear Random Forest classifier.

The Random Forest is the stronger deployment candidate:

| Model | Accuracy | Precision | Recall | ROC-AUC | CV ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.843 | 0.855 | 0.769 | 0.903 | 0.887 +/- 0.027 |
| Random Forest | 0.923 | 0.942 | 0.877 | 0.939 | 0.944 +/- 0.024 |

Business interpretation:

- Random Forest precision of 0.942 means most customers flagged for outreach are actual purchasers, reducing wasted campaign touches.
- Recall of 0.877 means the model captures most purchasers while still enforcing useful targeting discipline.
- In a simple targeting simulation, contacting the top 20% of customers by predicted purchase propensity produces a 98.3% conversion rate versus a 43.3% baseline rate, or a 2.27x conversion lift.
- Assuming $100 revenue per purchase, the top-20% targeting scenario identifies roughly 33 incremental purchases in the 300-customer holdout sample, or $3,300 in incremental revenue.

## Pipeline Architecture

The project is implemented as a reproducible scikit-learn workflow in [`src/main.py`](src/main.py):

1. Load customer-level data from [`data/customer_purchase_data.csv`](data/customer_purchase_data.csv).
2. Engineer an economic-value feature: `AvgSpendingPerPurchase`.
3. Split data into stratified train/test samples to preserve purchase-rate balance.
4. Apply preprocessing inside sklearn pipelines:
   - Median imputation and standard scaling for numeric features.
   - Most-frequent imputation and one-hot encoding for categorical features.
5. Train Logistic Regression and Random Forest classifiers with identical preprocessing contracts.
6. Validate model stability with 5-fold stratified cross-validation using ROC-AUC.
7. Evaluate holdout performance using accuracy, precision, recall, ROC-AUC, confusion matrices, and ROC curves.
8. Export business-labeled visualizations and Random Forest feature importances.

This structure avoids leakage because imputation, scaling, encoding, and feature creation are fit only on training folds during cross-validation and model training.

## Feature Engineering

| Feature | Type | Why It Matters |
|---|---|---|
| `TimeSpentOnWebsite` | Behavioral | Captures engagement intensity and near-term purchase intent. |
| `DiscountsAvailed` | Behavioral | Measures promotion sensitivity and offer responsiveness. |
| `LoyaltyProgram` | Behavioral / relationship | Indicates brand relationship depth and repeat-purchase potential. |
| `NumberOfPurchases` | Behavioral | Captures historical buying frequency. |
| `AvgSpendingPerPurchase` | Engineered | Normalizes income by purchase frequency to approximate customer economic value. |
| `AnnualIncome` | Demographic / capacity | Proxies purchasing power. |
| `Age`, `Gender`, `ProductCategory` | Demographic / preference | Provides secondary segmentation context. |

The model results show that behavioral indicators dominate demographic segmentation. For a business audience, the practical recommendation is to prioritize engagement, loyalty, and discount-response signals over broad demographic targeting.

## Model Results

### ROC Curve

The ROC curve compares each model's ability to rank likely purchasers above non-purchasers.

![ROC curve comparison](figures/roc_curve_comparison.png)

### Confusion Matrix

The Random Forest materially reduces both missed purchasers and wasted outreach compared with the baseline.

![Random Forest confusion matrix](figures/confusion_matrix_random_forest.png)

### Feature Importance

Random Forest feature importance identifies the strongest purchase drivers:

| Feature | Importance |
|---|---:|
| TimeSpentOnWebsite | 0.1856 |
| DiscountsAvailed | 0.1707 |
| Age | 0.1465 |
| LoyaltyProgram | 0.1432 |
| AnnualIncome | 0.1226 |
| NumberOfPurchases | 0.1013 |
| AvgSpendingPerPurchase | 0.0951 |
| ProductCategory | 0.0234 |
| Gender | 0.0116 |

![Random Forest feature importance](figures/feature_importance_random_forest.png)

## Business Decision Framing

This model is most useful as a campaign prioritization layer, not as a fully automated decision system.

Recommended deployment framing:

- Use predicted probabilities to rank customers, then select an outreach threshold based on budget.
- If campaign capacity is tight, optimize for precision to avoid spending on low-propensity customers.
- If the business goal is market coverage or new product launch awareness, lower the threshold to increase recall.
- Monitor conversion lift by decile so stakeholders can choose a targetable segment size with clear ROI.
- Recalibrate thresholds after each campaign because discount strategy, seasonality, and acquisition channels can shift purchase behavior.

## Visual Outputs

Generated figures are saved in [`figures/`](figures/):

- `roc_curve_comparison.png`
- `roc_curve_logistic_regression.png`
- `roc_curve_random_forest.png`
- `confusion_matrix_logistic.png`
- `confusion_matrix_random_forest.png`
- `feature_importance_random_forest.png`
- `feature_importance_random_forest.csv`
- `purchase_status_distribution.png`
- `boxplots_by_purchase_status.png`

## Interactive Dashboard

The repo also includes a Vercel-ready React dashboard in [`frontend/`](frontend/) that turns the ML outputs into an executive campaign targeting product.

![Dashboard preview](figures/dashboard_preview.png)

Dashboard features:

- KPI strip for ROC-AUC, precision, recall, conversion lift, and incremental revenue.
- Model comparison for Logistic Regression versus Random Forest.
- Interactive outreach-rate and revenue-per-purchase simulator.
- Business-labeled ROC curve, confusion matrix, feature importance, and segment strategy table.

## Run the Project

```bash
pip install -r requirements.txt
python src/main.py
```

The script prints model metrics, feature importances, and a business impact scenario, then regenerates all visual outputs in `figures/`.

Run the dashboard:

```bash
cd frontend
npm install
npm run dev
```

Build for Vercel:

```bash
cd frontend
npm run build
```

Vercel settings:

- Root directory: `frontend`
- Build command: `npm run build`
- Output directory: `dist`

## Portfolio Value

This project demonstrates end-to-end applied ML fluency:

- Business problem translation into a supervised learning objective.
- Leakage-safe preprocessing with sklearn pipelines.
- Baseline and nonlinear model comparison.
- Cross-validation plus holdout evaluation.
- Non-technical visual communication.
- Model interpretation connected to marketing strategy, conversion lift, and revenue impact.
- A deployable analytics dashboard that presents model results as a business decision tool.
