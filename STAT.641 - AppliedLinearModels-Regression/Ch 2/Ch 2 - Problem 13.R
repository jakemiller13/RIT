# Load packages
library(xlsx)

# Load data
air <- read.xlsx(paste('/Users/Jake/Google Drive/RIT/STAT.641 - ',
                       'AppliedLinearModels-Regression/data-prob-2-13.xlsx', 
                       sep = ''), 1)

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
cat('\nF_0 < F_statistic, DO NOT REJECT, NO LINEAR RELATIONSHIP')

### Problem 2.13.d - plot 95% CI bands ###
values <- data.frame(index = air$index)
ci_air = data.frame(predict(air.lm, values, interval = 'confidence',
                            level = 0.95))
ci_airlow <- ci_air[2]
ci_airhigh <- ci_air[3]

