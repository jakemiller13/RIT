# Load packages
library(xlsx)

# Load data
nrg <- read.xlsx(paste('/Users/Jake/Google Drive/RIT/STAT.641 - ',
                       'AppliedLinearModels-Regression/Ch 2/',
                       'data-table-B2.xlsx', sep = ''), 1)

# Create linear model
nrg.lm <- lm(y ~ x4, data = nrg)

# Plot heat flux (y) vs radial deflection (x4), add line
plot(y ~ x4, data = nrg)
abline(nrg.lm)

### Problem 2.3.a - LR model ###
coeffs <- coefficients(nrg.lm)
cat('\n--- Problem 2.3.a ---\n')
cat(coeffs)

### Problem 2.3.b - ANOVA ###
cat('\n\n--- Problem 2.3.b ---\n')
print(anova(nrg.lm))

### Problem 2.3.c - slope 99% CI ###
alpha <- 0.01
nu <- dim(nrg)[1] - 2
t <- qt(1 - alpha/2, nu)

slope <- summary(nrg.lm)$coefficients[2]
slope_se <- summary(nrg.lm)$coefficients[4]
ci_slopelow <- slope - slope_se * t
ci_slopehigh <- slope + slope_se * t
cat('\n--- Problem 2.3.c ---\n')
cat('99% CI:', ci_slopelow, 'to', ci_slopehigh)

### Problem 2.3.d - percent explained by model ###
cat('\n\n--- Problem 2.3.d ---\n')
cat('R-squared:', 100 * summary(nrg.lm)$r.squared, '%')

### Problem 2.3.e - 95% CI for 16.5 mrad ###
mrad <- data.frame(x4 = 16.5)
ci_mrad = predict(nrg.lm, mrad, interval = 'confidence', level = 0.95)
ci_mradlow <- ci_mrad[2]
ci_mradhigh <- ci_mrad[3]
cat('\n\n--- Problem 2.3.e ---\n')
cat('95% CI:', ci_mradlow, 'to', ci_mradhigh)