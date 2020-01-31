# Load packages
library(xlsx)
library(ggplot2)

# Load data
air <- read.xlsx(paste('/Users/Jake/Google Drive/RIT/STAT.641 - ',
                       'AppliedLinearModels-Regression/Ch 2/',
                       'data-prob-2-13.xlsx', sep = ''), 1)

# Create linear model
air.lm <- lm(days ~ index, data = air)

### Problem 2.13.a - Plot days vs index, add line ###
cat('\n--- Problem 2.13.a ---\n')
cat('    -> See plot <-')
plot(days ~ index, data = air)
abline(air.lm)

### Problem 2.13.b - equation ###
coeffs <- coefficients(air.lm)
cat('\n\n--- Problem 2.13.b ---\n')
cat(coeffs)

### Problem 2.13.c - significance of regression ###
f_statistic <- qf(0.95, 1, 16)
f_value <- anova(air.lm)$F[1]
p <- anova(air.lm)$Pr[1]
cat('\n\n--- Problem 2.13.c ---\n')
cat('F_0 =', f_value, '|', 'F_statistic =', f_statistic)
cat('\nF_0 < F_statistic, DO NOT REJECT, NO LINEAR RELATIONSHIP\n')

### Problem 2.13.d - plot 95% CI bands ###
cat('\n--- Problem 2.13.d ---\n')
cat('    -> See plot <-\n')
values <- data.frame(index = air$index)
ci_air = data.frame(predict(air.lm, values, interval = 'confidence',
                            level = 0.95))

# Prediction intervals
predictions <- predict(air.lm, interval = 'prediction')
df <- cbind(air, predictions)

# Confidence interval/prediction plot
ggplot(df, aes(x = index, y = days)) +
       geom_point() +
       geom_smooth(method = lm, se = TRUE) + 
       geom_line(aes(y = lwr), color = "red", linetype = "dashed") +
       geom_line(aes(y = upr), color = "red", linetype = "dashed")