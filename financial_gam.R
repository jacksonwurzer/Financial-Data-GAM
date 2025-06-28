# Generalized Additive Model to Analyze Financial Data
# author: Jackson Wurzer
# date: December 15th, 2024
# last update: June 2nd, 2025

# Load required libraries
library(mgcv)
library(Metrics)
library(ggplot2)

FinanceData <- read.csv("FinanceData.csv")

# Display column names
print("Column names:")
print(names(FinanceData))

# Create initial plots
plot(FinNetLending~Exports, data=FinanceData, 
     main='Exports vs. U.S. Financial Account Balance', pch=19, col='blue')

plot(FinNetLending~Imports, data=FinanceData, 
     main='Imports vs. U.S. Financial Account Balance', pch=19, col='red')

plot(FinNetLending~DirectPortfolioLiabilities, data=FinanceData, 
     main='Direct Portfolio Liabilities vs. U.S. Financial Account Balance', pch=19, col='orange')

# Set seed for reproducibility
set.seed(123)

# Split data into training/validation/test sets (60%/20%/20%)
trainIndices <- sample(1:nrow(FinanceData), size = 0.6 * nrow(FinanceData))
trainData <- FinanceData[trainIndices, ]

remainingIndices <- setdiff(1:nrow(FinanceData), trainIndices)
validationIndices <- sample(remainingIndices, size = 0.5 * length(remainingIndices))
validationData <- FinanceData[validationIndices, ]

testIndices <- setdiff(remainingIndices, validationIndices)
testData <- FinanceData[testIndices, ]

print(paste("Training set size:", nrow(trainData)))
print(paste("Validation set size:", nrow(validationData)))
print(paste("Test set size:", nrow(testData)))

# Fit GAM models
# Full model including all predictors
fullGam <- gam(FinNetLending ~ s(Exports) + s(Imports) + s(DirectPortfolioLiabilities) + 
                 s(Period) + s(DirectInvestmentLiabilities) + s(ReserveAssets) + 
                 s(DirectInvestmentAssets), data=trainData)
summary(fullGam)

# First reduced model removing some variables
reducedGam1 <- gam(FinNetLending ~ s(Exports) + s(Imports) + s(DirectInvestmentAssets) + 
                     s(DirectPortfolioLiabilities), data=trainData)
summary(reducedGam1)

# Final reduced model with only three key predictors
finalReducedGam <- gam(FinNetLending ~ s(Exports) + s(Imports) + s(DirectPortfolioLiabilities), 
                       data=trainData)

print("Summary of final reduced GAM:")
summary(finalReducedGam)

# Validation model with tuned number of basis functions (k)
validationGam <- gam(FinNetLending ~ s(Exports, k=6) + s(Imports, k=9) + 
                       s(DirectPortfolioLiabilities, k=4), data=validationData)

print("Summary of validation GAM:")
summary(validationGam)

validationPrediction <- predict(validationGam, newdata=validationData)
validation_rmse <- rmse(validationData$FinNetLending, validationPrediction)
print(paste("Validation RMSE:", validation_rmse))

# Test model with same specifications as validation
testGam <- gam(FinNetLending ~ s(Exports, k=6) + s(Imports, k=9) + 
                 s(DirectPortfolioLiabilities, k=4), data=testData)

print("Summary of test GAM:")
summary(testGam)

testPrediction <- predict(testGam, newdata=testData)
test_rmse <- rmse(testData$FinNetLending, testPrediction)
print(paste("Test RMSE:", test_rmse))

# Individual variable analysis - Exports
exportsGam <- gam(FinNetLending ~ s(Exports, k=6), data=testData)

# Create sequence of export values for smooth prediction line
exportSeq <- data.frame(Exports = seq(min(testData$Exports), max(testData$Exports), length.out = 100))

# Generate predictions with standard errors for confidence intervals
predictions <- predict(exportsGam, newdata = exportSeq, type = "response", se.fit = TRUE)
exportSeq$fit <- predictions$fit
exportSeq$lower <- predictions$fit - 1.96 * predictions$se.fit  # 95% confidence interval
exportSeq$upper <- predictions$fit + 1.96 * predictions$se.fit

# Visualize predictions using Exports
plot1 <- ggplot(testData, aes(x = Exports, y = FinNetLending)) + 
  geom_point(color = "blue", alpha = 0.7, size = 2) +  
  geom_line(data = exportSeq, aes(x = Exports, y = fit), color = "maroon", linewidth = 0.7) + 
  geom_ribbon(data = exportSeq, aes(x = Exports, ymin = lower, ymax = upper), fill = "maroon", alpha = 0.2) + 
  labs(title = "Exports vs. Financial Account Balance", 
       x = "Exports", 
       y = "Financial Account Balance") +
  theme_minimal()
print(plot1)

# Individual variable analysis - Imports
importsGam <- gam(FinNetLending ~ s(Imports, k=9), data=testData)

# Create sequence of import values for smooth prediction line
importSeq <- data.frame(Imports = seq(min(testData$Imports), max(testData$Imports), length.out = 100))

# Generate predictions with standard errors for confidence intervals
predictions <- predict(importsGam, newdata = importSeq, type = "response", se.fit = TRUE)
importSeq$fit <- predictions$fit
importSeq$lower <- predictions$fit - 1.96 * predictions$se.fit  # 95% confidence interval
importSeq$upper <- predictions$fit + 1.96 * predictions$se.fit

# Visualize predictions using Imports
plot2 <- ggplot(testData, aes(x = Imports, y = FinNetLending)) + 
  geom_point(color = "red", alpha = 0.7, size = 2) +  
  geom_line(data = importSeq, aes(x = Imports, y = fit), color = "black", linewidth = 0.7) + 
  geom_ribbon(data = importSeq, aes(x = Imports, ymin = lower, ymax = upper), fill = "black", alpha = 0.2) + 
  labs(title = "Imports vs. Financial Account Balance", 
       x = "Imports", 
       y = "Financial Account Balance") +
  theme_minimal()
print(plot2)

# Individual variable analysis - Direct Portfolio Liabilities
portfolioGam <- gam(FinNetLending ~ s(DirectPortfolioLiabilities, k=4), data = testData)

# Create sequence of portfolio liability values for smooth prediction line
portfolioSeq <- data.frame(DirectPortfolioLiabilities = seq(min(testData$DirectPortfolioLiabilities), 
                                                            max(testData$DirectPortfolioLiabilities), length.out = 100))

# Generate predictions with standard errors for confidence intervals
predictions <- predict(portfolioGam, newdata = portfolioSeq, type = "response", se.fit = TRUE)
portfolioSeq$fit <- predictions$fit
portfolioSeq$lower <- predictions$fit - 1.96 * predictions$se.fit  # 95% confidence interval
portfolioSeq$upper <- predictions$fit + 1.96 * predictions$se.fit

# Visualize predictions using Direct Portfolio Liabilities
plot3 <- ggplot(testData, aes(x = DirectPortfolioLiabilities, y = FinNetLending)) + 
  geom_point(color = "orange", alpha = 0.7, size = 2) +  
  geom_line(data = portfolioSeq, aes(x = DirectPortfolioLiabilities, y = fit), color = "green", linewidth = 0.7) + 
  geom_ribbon(data = portfolioSeq, aes(x = DirectPortfolioLiabilities, ymin = lower, ymax = upper), fill = "green", alpha = 0.2) + 
  labs(title = "Direct Portfolio Liabilities vs. Financial Account Balance", 
       x = "Direct Portfolio Liabilities", 
       y = "Financial Account Balance") +
  theme_minimal()
print(plot3)

# Final comparison plot - Actual vs Predicted
# Apply test model to entire dataset for comprehensive evaluation
fullDataPrediction <- predict(testGam, newdata = FinanceData)
comparisonData <- data.frame(Actual = FinanceData$FinNetLending, Predicted = fullDataPrediction)

finalPlot <- ggplot(comparisonData, aes(x = Actual, y = Predicted)) +
  geom_point(color = "red", alpha = 0.7, size = 2) +  
  geom_abline(slope = 1, intercept = 0, color = "black", linetype = "dashed") + 
  labs(title = "Actual vs. Predicted Financial Account Balance",
       x = "Actual Financial Account Balance",
       y = "Predicted Financial Account Balance") +
  theme_minimal()

print(finalPlot)

# Model Conditions
print("Creating plots to check model conditions...")

# Q-Q plot for normality of residuals
qqnorm(resid(testGam), main="Q-Q Plot of Residuals")
qqline(resid(testGam))

# Residuals vs Fitted plot to check for heteroscedasticity 
plot(fitted(testGam), resid(testGam), 
     main="Residuals vs. Fitted", 
     xlab="Fitted Values", 
     ylab="Residuals",
     pch=19, col="blue")
abline(h=0, col="red", lty=2)  # Add horizontal reference line at zero

print("Analysis complete!")