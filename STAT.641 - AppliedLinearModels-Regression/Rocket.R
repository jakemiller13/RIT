# Rocket Prop Example: Simple Linear Regress Chapter 2
# Y response variable: shear strength
# X predictor var.   : age

# set up the working director
setwd("/Users/Jake/Google Drive/RIT/STAT.641 - AppliedLinearModels-Regression")

# read in your data
library(xlsx)
rocket <- read.xlsx("Rocket_Prop.xls",1)
rocket

# scatter plot
plot(strength ~ age, data=rocket)

# Simple Linear Regression Analysis

rocket.lm <- lm(strength ~ age, data=rocket)
rocket.lm

# add a regression line to the plot
abline(rocket.lm)

#coefficents
coeffs <- coefficients(rocket.lm)
coeffs

#summory of rocket.lm
summary(rocket.lm)

#fitted values obtained using the function fitted()
#rediduals obtained using the function resid()
# create a table with the fitted values and residuals

roc.data <- data.frame(rocket, fitted.value=fitted(rocket.lm), residual=resid(rocket.lm))
roc.data

# anova (analysis of variance)
anova(rocket.lm)

# when age=16
newdata <- data.frame(age=16)
#confidence interval of mean responce
predict(rocket.lm, newdata, interval="confidence", level=0.95)
#prediction interval
predict(rocket.lm, newdata, interval="prediction")



