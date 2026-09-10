# Customer Churn Analysis

Analysis and prediction of customer churn using the real IBM Telco Customer Churn dataset (7,043 customers).

## Project Highlights
- Exploratory Data Analysis with multiple visualizations
- Random Forest classification model
- **Accuracy: 80.24%**
- **AUC: 0.843**

## Key Findings
- Month to-month contracts have the highest churn rate (~43%)
- Fiber optic customers churn significantly more than DSL
- Electronic check payment method is strongly associated with churn
- Customers with shorter tenure and higher monthly charges are more likely to leave

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
