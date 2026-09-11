# Customer Churn Analysis
# Dataset: IBM Telco Customer Churn

library(randomForest)
library(pROC)
library(data.table)


# 1. Load data

churn <- fread("Telco-Customer-Churn.csv")

cat("Original rows:", nrow(churn), "\n")


# 2. Clean the data

churn[, TotalCharges := as.numeric(TotalCharges)]

# Remove missing values
churn <- churn[!is.na(TotalCharges)]

# Change character columns to factors
char_cols <- names(churn)[sapply(churn, is.character)]

for (col in char_cols) {
  churn[[col]] <- as.factor(churn[[col]])
}

# Change SeniorCitizen to Yes/No
churn[, SeniorCitizen := factor(
  SeniorCitizen,
  levels = c(0, 1),
  labels = c("No", "Yes")
)]

# Customer ID is not useful for prediction
churn[, customerID := NULL]

cat("Cleaned rows:", nrow(churn), "\n")

cat("\nChurn distribution:\n")
print(table(churn$Churn))

churn_rate <- mean(churn$Churn == "Yes") * 100
cat("\nOverall Churn Rate:", round(churn_rate, 1), "%\n")


# 3. Basic analysis

cat("\n=== KEY INSIGHTS ===\n")

cat("\nChurn rate by Contract:\n")
print(prop.table(table(churn$Contract, churn$Churn), 1) * 100)

cat("\nChurn rate by Internet Service:\n")
print(prop.table(table(churn$InternetService, churn$Churn), 1) * 100)

cat("\nChurn rate by Payment Method:\n")
print(prop.table(table(churn$PaymentMethod, churn$Churn), 1) * 100)

cat("\nAverage Tenure:\n")
print(aggregate(tenure ~ Churn, data = churn, mean))

cat("\nAverage Monthly Charges:\n")
print(aggregate(MonthlyCharges ~ Churn, data = churn, mean))


# 4. Visualisations

# Churn distribution
png("01_churn_distribution.png", width = 800, height = 600)

barplot(
  table(churn$Churn),
  col = c("#2E86AB", "#E94F37"),
  main = "Customer Churn Distribution",
  ylab = "Number of Customers",
  xlab = "Churn"
)

dev.off()


# Churn by contract
png("02_churn_by_contract.png", width = 900, height = 600)

contract_table <- prop.table(
  table(churn$Contract, churn$Churn),
  1
)

barplot(
  t(contract_table),
  beside = TRUE,
  col = c("#2E86AB", "#E94F37"),
  main = "Churn Rate by Contract Type",
  ylab = "Proportion",
  legend.text = colnames(contract_table),
  args.legend = list(x = "topright")
)

dev.off()


# Churn by internet service
png("03_churn_by_internet.png", width = 900, height = 600)

internet_table <- prop.table(
  table(churn$InternetService, churn$Churn),
  1
)

barplot(
  t(internet_table),
  beside = TRUE,
  col = c("#2E86AB", "#E94F37"),
  main = "Churn Rate by Internet Service",
  ylab = "Proportion",
  legend.text = colnames(internet_table),
  args.legend = list(x = "topright")
)

dev.off()


# Tenure by churn
png("04_tenure_by_churn.png", width = 800, height = 600)

boxplot(
  tenure ~ Churn,
  data = churn,
  col = c("#2E86AB", "#E94F37"),
  main = "Tenure by Churn Status",
  ylab = "Tenure (months)",
  xlab = "Churn"
)

dev.off()


# Monthly charges by churn
png("05_monthlycharges_by_churn.png", width = 800, height = 600)

boxplot(
  MonthlyCharges ~ Churn,
  data = churn,
  col = c("#2E86AB", "#E94F37"),
  main = "Monthly Charges by Churn Status",
  ylab = "Monthly Charges ($)",
  xlab = "Churn"
)

dev.off()


# Churn by payment method
png("06_churn_by_payment.png", width = 1000, height = 600)

payment_table <- prop.table(
  table(churn$PaymentMethod, churn$Churn),
  1
)

barplot(
  t(payment_table),
  beside = TRUE,
  col = c("#2E86AB", "#E94F37"),
  main = "Churn Rate by Payment Method",
  ylab = "Proportion",
  legend.text = colnames(payment_table),
  args.legend = list(x = "topright"),
  las = 2,
  cex.names = 0.8
)

dev.off()

cat("\n6 plots saved\n")


# 5. Split data into training and testing

set.seed(42)

n <- nrow(churn)

train_rows <- sample(
  1:n,
  size = floor(0.8 * n)
)

train_data <- churn[train_rows, ]
test_data <- churn[-train_rows, ]


# 6. Random Forest

cat("\n=== TRAINING RANDOM FOREST ===\n")

rf_model <- randomForest(
  Churn ~ .,
  data = train_data,
  ntree = 200,
  importance = TRUE,
  mtry = 4
)

print(rf_model)


# Make predictions

rf_pred <- predict(
  rf_model,
  newdata = test_data
)

rf_prob <- predict(
  rf_model,
  newdata = test_data,
  type = "prob"
)[, "Yes"]


# Confusion matrix

cat("\n=== CONFUSION MATRIX ===\n")

cm <- table(
  Predicted = rf_pred,
  Actual = test_data$Churn
)

print(cm)


# Accuracy

accuracy <- sum(diag(cm)) / sum(cm)

cat("\nAccuracy:", round(accuracy * 100, 2), "%\n")


# Precision, recall and F1 score

tp <- cm["Yes", "Yes"]
fp <- cm["Yes", "No"]
fn <- cm["No", "Yes"]

precision <- tp / (tp + fp)
recall <- tp / (tp + fn)

f1 <- 2 * precision * recall / (precision + recall)

cat("Precision:", round(precision, 3), "\n")
cat("Recall:   ", round(recall, 3), "\n")
cat("F1 Score: ", round(f1, 3), "\n")


# 7. ROC curve

roc_result <- roc(
  test_data$Churn,
  rf_prob,
  levels = c("No", "Yes")
)

auc_value <- auc(roc_result)

cat("\nAUC:", round(auc_value, 3), "\n")

png("07_roc_curve.png", width = 800, height = 600)

plot(
  roc_result,
  main = paste(
    "ROC Curve - Random Forest (AUC =",
    round(auc_value, 3),
    ")"
  ),
  col = "#E94F37",
  lwd = 3
)

abline(
  a = 0,
  b = 1,
  lty = 2,
  col = "gray"
)

dev.off()


# 8. Feature importance

png("08_feature_importance.png", width = 900, height = 700)

varImpPlot(
  rf_model,
  main = "Feature Importance - Random Forest",
  col = "#2E86AB"
)

dev.off()


# 9. Save results

sink("model_results.txt")

cat("============================================================\n")
cat("CUSTOMER CHURN ANALYSIS - MODEL RESULTS\n")
cat("Dataset: IBM Telco Customer Churn\n")
cat("============================================================\n\n")

cat(
  "Overall Churn Rate:",
  round(churn_rate, 1),
  "%\n\n"
)

cat("--- RANDOM FOREST PERFORMANCE ---\n")

cat(
  "Accuracy:  ",
  round(accuracy * 100, 2),
  "%\n"
)

cat(
  "Precision: ",
  round(precision, 3),
  "\n"
)

cat(
  "Recall:    ",
  round(recall, 3),
  "\n"
)

cat(
  "F1 Score:  ",
  round(f1, 3),
  "\n"
)

cat(
  "AUC:       ",
  round(auc_value, 3),
  "\n\n"
)

cat("--- CONFUSION MATRIX ---\n")
print(cm)

cat("\n--- TOP FEATURES ---\n")

feature_importance <- importance(rf_model)

feature_importance <- feature_importance[
  order(-feature_importance[, "MeanDecreaseAccuracy"]),
]

print(feature_importance[1:10, ])


cat("\n============================================================\n")
cat("KEY BUSINESS INSIGHTS\n")
cat("============================================================\n")

cat("- Month-to-month contracts have higher churn\n")
cat("- Fiber optic customers have higher churn than DSL customers\n")
cat("- Electronic check users have higher churn\n")
cat("- Customers with shorter tenure are more likely to leave\n")
cat("- Higher monthly charges are associated with higher churn\n")

cat("============================================================\n")

sink()


# Finished

cat("\nAll files generated successfully!\n")

cat("Files created:\n")
cat("- 01_churn_distribution.png\n")
cat("- 02_churn_by_contract.png\n")
cat("- 03_churn_by_internet.png\n")
cat("- 04_tenure_by_churn.png\n")
cat("- 05_monthlycharges_by_churn.png\n")
cat("- 06_churn_by_payment.png\n")
cat("- 07_roc_curve.png\n")
cat("- 08_feature_importance.png\n")
cat("- model_results.txt\n")
