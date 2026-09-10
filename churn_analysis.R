# ============================================================
# Customer Churn Analysis - Complete Project (Base R version)
# Dataset: IBM Telco Customer Churn (real data)
# ============================================================

library(randomForest)
library(pROC)
library(data.table)

# ----------------------------
# 1. LOAD DATA
# ----------------------------
churn <- fread("Telco-Customer-Churn.csv")

cat("Original rows:", nrow(churn), "\n")

# ----------------------------
# 2. DATA CLEANING
# ----------------------------
# Convert TotalCharges to numeric
churn[, TotalCharges := as.numeric(TotalCharges)]

# Remove rows with missing TotalCharges
churn <- churn[!is.na(TotalCharges)]

# Convert character columns to factors
char_cols <- names(churn)[sapply(churn, is.character)]
for (col in char_cols) {
  churn[[col]] <- as.factor(churn[[col]])
}

# SeniorCitizen to factor
churn[, SeniorCitizen := factor(SeniorCitizen, levels = c(0, 1), labels = c("No", "Yes"))]

# Remove customerID
churn[, customerID := NULL]

cat("Cleaned rows:", nrow(churn), "\n")
cat("\nChurn distribution:\n")
print(table(churn$Churn))
cat("\nOverall Churn Rate:", round(mean(churn$Churn == "Yes") * 100, 1), "%\n")

# ----------------------------
# 3. BASIC EDA SUMMARIES
# ----------------------------
cat("\n=== KEY INSIGHTS ===\n")

# Churn by Contract
cat("\nChurn rate by Contract:\n")
print(prop.table(table(churn$Contract, churn$Churn), 1) * 100)

# Churn by InternetService
cat("\nChurn rate by Internet Service:\n")
print(prop.table(table(churn$InternetService, churn$Churn), 1) * 100)

# Churn by PaymentMethod
cat("\nChurn rate by Payment Method:\n")
print(prop.table(table(churn$PaymentMethod, churn$Churn), 1) * 100)

# Average tenure and charges
cat("\nAverage Tenure (Churned vs Stayed):\n")
print(aggregate(tenure ~ Churn, data = churn, mean))

cat("\nAverage MonthlyCharges (Churned vs Stayed):\n")
print(aggregate(MonthlyCharges ~ Churn, data = churn, mean))

# ----------------------------
# 4. VISUALIZATIONS (Base R)
# ----------------------------

# Plot 1: Churn distribution
png("01_churn_distribution.png", width = 800, height = 600)
barplot(table(churn$Churn),
        col = c("#2E86AB", "#E94F37"),
        main = "Customer Churn Distribution",
        ylab = "Number of Customers",
        xlab = "Churn Status")
dev.off()

# Plot 2: Churn by Contract
png("02_churn_by_contract.png", width = 900, height = 600)
contract_tab <- prop.table(table(churn$Contract, churn$Churn), 1)
barplot(t(contract_tab),
        beside = TRUE,
        col = c("#2E86AB", "#E94F37"),
        main = "Churn Rate by Contract Type",
        ylab = "Proportion",
        legend.text = colnames(contract_tab),
        args.legend = list(x = "topright"))
dev.off()

# Plot 3: Churn by Internet Service
png("03_churn_by_internet.png", width = 900, height = 600)
internet_tab <- prop.table(table(churn$InternetService, churn$Churn), 1)
barplot(t(internet_tab),
        beside = TRUE,
        col = c("#2E86AB", "#E94F37"),
        main = "Churn Rate by Internet Service",
        ylab = "Proportion",
        legend.text = colnames(internet_tab),
        args.legend = list(x = "topright"))
dev.off()

# Plot 4: Tenure boxplot
png("04_tenure_by_churn.png", width = 800, height = 600)
boxplot(tenure ~ Churn, data = churn,
        col = c("#2E86AB", "#E94F37"),
        main = "Tenure by Churn Status",
        ylab = "Tenure (months)",
        xlab = "Churn")
dev.off()

# Plot 5: Monthly Charges boxplot
png("05_monthlycharges_by_churn.png", width = 800, height = 600)
boxplot(MonthlyCharges ~ Churn, data = churn,
        col = c("#2E86AB", "#E94F37"),
        main = "Monthly Charges by Churn Status",
        ylab = "Monthly Charges ($)",
        xlab = "Churn")
dev.off()

# Plot 6: Payment Method
png("06_churn_by_payment.png", width = 1000, height = 600)
payment_tab <- prop.table(table(churn$PaymentMethod, churn$Churn), 1)
barplot(t(payment_tab),
        beside = TRUE,
        col = c("#2E86AB", "#E94F37"),
        main = "Churn Rate by Payment Method",
        ylab = "Proportion",
        legend.text = colnames(payment_tab),
        args.legend = list(x = "topright"),
        las = 2,
        cex.names = 0.8)
dev.off()

cat("\n✔ 6 plots saved\n")

# ----------------------------
# 5. PREPARE FOR MODELING
# ----------------------------
set.seed(42)

# Train/test split (80/20)
n <- nrow(churn)
train_idx <- sample(1:n, size = floor(0.8 * n))
train_data <- churn[train_idx, ]
test_data  <- churn[-train_idx, ]

# ----------------------------
# 6. RANDOM FOREST MODEL
# ----------------------------
cat("\n=== TRAINING RANDOM FOREST ===\n")

rf_model <- randomForest(Churn ~ ., data = train_data,
                         ntree = 200,
                         importance = TRUE,
                         mtry = 4)

print(rf_model)

# Predictions
rf_pred <- predict(rf_model, newdata = test_data)
rf_prob <- predict(rf_model, newdata = test_data, type = "prob")[, "Yes"]

# Confusion Matrix
cat("\n=== CONFUSION MATRIX (Random Forest) ===\n")
cm <- table(Predicted = rf_pred, Actual = test_data$Churn)
print(cm)

accuracy <- sum(diag(cm)) / sum(cm)
cat("\nAccuracy:", round(accuracy * 100, 2), "%\n")

# Precision, Recall, F1 for "Yes" class
tp <- cm["Yes", "Yes"]
fp <- cm["Yes", "No"]
fn <- cm["No", "Yes"]
precision <- tp / (tp + fp)
recall <- tp / (tp + fn)
f1 <- 2 * precision * recall / (precision + recall)

cat("Precision (Churn):", round(precision, 3), "\n")
cat("Recall (Churn):   ", round(recall, 3), "\n")
cat("F1 Score:         ", round(f1, 3), "\n")

# ----------------------------
# 7. ROC CURVE
# ----------------------------
roc_obj <- roc(test_data$Churn, rf_prob, levels = c("No", "Yes"))
auc_value <- auc(roc_obj)

cat("\nAUC:", round(auc_value, 3), "\n")

png("07_roc_curve.png", width = 800, height = 600)
plot(roc_obj, main = paste0("ROC Curve - Random Forest (AUC = ", round(auc_value, 3), ")"),
     col = "#E94F37", lwd = 3)
abline(a = 0, b = 1, lty = 2, col = "gray")
dev.off()

# ----------------------------
# 8. FEATURE IMPORTANCE
# ----------------------------
png("08_feature_importance.png", width = 900, height = 700)
varImpPlot(rf_model,
           main = "Feature Importance (Random Forest)",
           col = "#2E86AB")
dev.off()

# ----------------------------
# 9. SAVE SUMMARY RESULTS
# ----------------------------
sink("model_results.txt")
cat("============================================================\n")
cat("CUSTOMER CHURN ANALYSIS - MODEL RESULTS\n")
cat("Dataset: IBM Telco Customer Churn (real data)\n")
cat("============================================================\n\n")

cat("Overall Churn Rate:", round(mean(churn$Churn == "Yes") * 100, 1), "%\n\n")

cat("--- RANDOM FOREST PERFORMANCE ---\n")
cat("Accuracy:  ", round(accuracy * 100, 2), "%\n")
cat("Precision: ", round(precision, 3), "\n")
cat("Recall:    ", round(recall, 3), "\n")
cat("F1 Score:  ", round(f1, 3), "\n")
cat("AUC:       ", round(auc_value, 3), "\n\n")

cat("--- CONFUSION MATRIX ---\n")
print(cm)

cat("\n--- TOP FEATURES (Mean Decrease Accuracy) ---\n")
print(importance(rf_model)[order(-importance(rf_model)[, "MeanDecreaseAccuracy"]), ][1:10, ])

cat("\n============================================================\n")
cat("Key Business Insights:\n")
cat("- Month-to-month contracts have significantly higher churn\n")
cat("- Fiber optic customers churn more than DSL\n")
cat("- Electronic check payment method is associated with higher churn\n")
cat("- Customers with shorter tenure are more likely to leave\n")
cat("- Higher monthly charges correlate with higher churn risk\n")
cat("============================================================\n")
sink()

cat("\n✔ All files generated successfully!\n")
cat("Files created:\n")
cat("  - 01_churn_distribution.png\n")
cat("  - 02_churn_by_contract.png\n")
cat("  - 03_churn_by_internet.png\n")
cat("  - 04_tenure_by_churn.png\n")
cat("  - 05_monthlycharges_by_churn.png\n")
cat("  - 06_churn_by_payment.png\n")
cat("  - 07_roc_curve.png\n")
cat("  - 08_feature_importance.png\n")
cat("  - model_results.txt\n")
