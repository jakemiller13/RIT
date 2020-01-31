# Load packages
library(xlsx)

# Load data
nfl <- read.xlsx(paste('/Users/Jake/Google Drive/RIT/STAT.641 - ',
                       'AppliedLinearModels-Regression/Ch 2/',
                       'data-table-B1.xlsx', sep = ''), 1)

# Create linear model
nfl.lm <- lm(y ~ x8, data = nfl)

# Plot games won (y) vs yards gained rushing by opponents (x8), add line
plot(y ~ x8, data = nfl)
abline(nfl.lm)

### Problem 2.1.a - LR model ###
coeffs <- coefficients(nfl.lm)
cat('\n--- Problem 2.1.a ---\n')
cat(coeffs)

### Problem 2.1.b - ANOVA ###
cat('\n\n--- Problem 2.1.b ---\n')
print(anova(nfl.lm))

### Problem 2.1.c - slope 95% CI ###
alpha <- 0.05
nu <- dim(nfl)[1] - 2
t <- qt(1 - alpha/2, nu)

slope <- summary(nfl.lm)$coefficients[2]
slope_se <- summary(nfl.lm)$coefficients[4]
ci_slopelow <- slope - slope_se * t
ci_slopehigh <- slope + slope_se * t
cat('\n--- Problem 2.1.c ---\n')
cat('95% CI:', ci_slopelow, 'to', ci_slopehigh)

### Problem 2.1.d - percent explained by model ###
cat('\n\n--- Problem 2.1.d ---\n')
cat('R-squared:', 100 * summary(nfl.lm)$r.squared, '%')

### Problem 2.1.e - 95% ci for 2000 yards ###
yards <- data.frame(x8 = 2000)
ci_yards = predict(nfl.lm, yards, interval = 'confidence', level = 0.95)
ci_yardslow <- ci_yards[2]
ci_yardshigh <- ci_yards[3]
cat('\n\n--- Problem 2.1.e ---\n')
cat('95% CI:', ci_yardslow, 'to', ci_yardshigh)