# Customer Churn Analysis

Analysis and prediction of customer churn using the real IBM Telco Customer Churn dataset (7,043 customers).

## Project Highlights
- Exploratory Data Analysis with multiple visualizations
- Random Forest classification model
- **Accuracy: 80.24%**
- **AUC: 0.843**

## Key Findings
- Month-to-month contracts have the highest churn rate (~43%)
- Fiber optic customers churn significantly more than DSL
- Electronic check payment method is strongly associated with churn
- Customers with shorter tenure and higher monthly charges are more likely to leave

## Machine Learning Model

**Algorithm:** Random Forest (200 decision trees)

**Why Random Forest?** It handles mixed categorical/numeric data well, resists
overfitting, and gives feature importance scores — useful for explaining *which*
factors drive churn, not just predicting it.

**Pipeline:**
1. Data split 80% train / 20% test (stratified on churn)
2. Model trained on 15 customer features (contract, tenure, charges, services, etc.)
3. Evaluated on the held-out test set

**Performance:**

| Metric | Score |
|--------|-------|
| Accuracy | 80.24% |
| AUC | 0.843 |
| Precision (Churn) | 0.655 |
| Recall (Churn) | 0.543 |
| F1 Score | 0.594 |

**Top predictive features:** tenure, TotalCharges, MonthlyCharges, Contract type,
and InternetService — confirming what the EDA showed.

## Files
| File | Description |
|------|-------------|
| `churn_analysis.R` | Full analysis script |
| `Telco-Customer-Churn.csv` | Original dataset |
| `model_results.txt` | Model performance summary |
| `01–08_*.png` | Visualizations and ROC curve |

## How to Run
```r
# Open churn_analysis.R in RStudio and run the entire script
