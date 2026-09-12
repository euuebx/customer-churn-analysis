# Customer Churn Analysis

Predicts which telecom customers are likely to leave, using R with Random Forest.

## Dataset

IBM Telco Customer Churn:
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

File: `Telco-Customer-Churn.csv`. About 7,000 customers, 26.6% churned.

## Results

Random Forest (200 trees, mtry = 4), 80/20 train/test split.

| Metric | Value |
|---|---|
| Accuracy | 80.24% |
| Precision | 0.655 |
| Recall | 0.543 |
| F1 | 0.594 |
| AUC | 0.843 |

Note: always predicting "no churn" would give about 73% accuracy since only 26.6% churned. Recall of 0.543 means the model catches about half of actual churners.

## Observations

Patterns in the data, not causes.

- Month-to-month churns at 42.7%, one year 11.3%, two year 2.8%
- Fiber optic churns at 41.9%, DSL 19.0%, no internet 7.4%
- Electronic check churns at 45.3%, other methods 15 to 19%
- Churners had shorter tenure (18 vs 37.7 months) and higher monthly charges ($74.44 vs $61.31)

## Files

| File | What it is |
|---|---|
| `churn_analysis.R` | main script |
| `Telco-Customer-Churn.csv` | dataset |
| `01` to `06` PNGs | EDA plots |
| `07_roc_curve.png` | ROC curve |
| `08_feature_importance.png` | feature importance |
| `model_results.txt` | full results |

## Requirements

```r
install.packages(c("randomForest", "pROC", "data.table"))
```

## How to run

From the project folder in PowerShell:

```powershell
& "C:\Program Files\R\R-4.5.2\bin\Rscript.exe" churn_analysis.R
```

Adjust the R version if yours is different.
