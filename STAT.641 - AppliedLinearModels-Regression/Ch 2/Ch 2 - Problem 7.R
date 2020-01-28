# Load packages
library(xlsx)

# Load data
oxy <- read.xlsx(paste('/Users/Jake/Google Drive/RIT/STAT.641 - ',
                       'AppliedLinearModels-Regression/data-prob-2-7.xls', 
                       sep = ''), 1)

# Create linear model
oxy.lm <- lm(purity ~ hydro, data = oxy)

# Plot purity (y) vs hydrocarbon (x4), add line
plot(purity ~ hydro, data = oxy)
abline(oxy.lm)

### Problem 2.7.a - LR model ###
coeffs <- coefficients(oxy.lm)
cat('\n--- Problem 2.7.a ---\n')
cat(coeffs)

### Problem 2.7.b - H0: B1 = 0 ###
slope <- summary(oxy.lm)$coefficients[2]
slope_se <- summary(oxy.lm)$coefficients[4]
t0 <- slope/slope_se

alpha <- 0.05
nu <- dim(oxy)[1] - 2
crit_t <- qt(1 - alpha/2, nu)

cat('\n\n--- Problem 2.7.b ---\n')
cat('t0 =', t0, '>', 'critical_t =', crit_t)
cat('\n-> Reject H0 <-')

### Problem 2.7.c - R-squared ###
cat('\n\n--- Problem 2.7.c ---\n')
cat('R-squared:', 100 * summary(oxy.lm)$r.squared, '%')

### Problem 2.7.d - slope 95% CI ###
t <- qt(1 - alpha/2, nu)
ci_slopelow <- slope - slope_se * t
ci_slopehigh <- slope + slope_se * t
cat('\n\n--- Problem 2.7.d ---\n')
cat('95% CI:', ci_slopelow, 'to', ci_slopehigh)

### Problem 2.7.e - 95% CI for 1.00% hydrocarbon ###
hyd <- data.frame(hydro = 1.00)
ci_hyd = predict(oxy.lm, hyd, interval = 'confidence', level = 0.95)
ci_hydlow <- ci_hyd[2]
ci_hydhigh <- ci_hyd[3]
cat('\n\n--- Problem 2.3.e ---\n')
cat('95% CI:', ci_hydlow, 'to', ci_hydhigh)