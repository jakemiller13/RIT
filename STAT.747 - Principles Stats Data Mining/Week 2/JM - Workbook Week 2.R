# Load libraries - mlbench still not working
library(class)
library(MASS)
library(kernlab)
#library(mlbench)
library(reshape2)
library(ROCR)
library(ggplot2)

# Load data
xy <- '/Users/Jake/Google Drive/RIT/STAT.747 - Principles Stats Data Mining/Data/PimaIndiansDiabetes.csv'        # Store data in xy frame
xy <- Pima.tr

# First/last 6
head(xy)
tail(xy)

# Data types
str(xy)

# Latex table of data
library(xtable)
xtable(head(xy))

# Set dimensions
n   <- nrow(xy)       # Sample size
p   <- ncol(xy) - 1   # Dimensionality of the input space
pos <- p+1            # Position of the response
x   <- xy[,-pos]      # Data matrix: n x p matrix
y   <- xy[, pos]      # Response vector

# Correlation plot
library(corrplot)
corrplot(cor(x))

# Pairwise plots
plot(x, col = as.numeric(y) + 2)

# Each variable alone
par(mfrow = c(3, 3))
for(j in 1:p)
{
  boxplot(x[,j]~y, col = 2:3, ylab = colnames(x)[j], xlab = 'diabetic')
}

# Look at all variables
par(mfrow = c(1, 1))
boxplot(x)

# Distribution of response
par(mfrow = c(1, 1))
barplot(prop.table(table(y)), col = 2:3, xlab = 'diabetic')

# Change labels for ROC
xy[,pos]  <- as.factor(ifelse(xy[,pos] == unique(xy[,pos])[1], 1, 0))
y         <- as.factor(ifelse(y == unique(y)[1], 1, 0))

# Split data
set.seed (19671210)   # Set seed for random number generation to be reproducible

epsilon <- 1/3               # Proportion of observations in the test set
nte     <- round(n * epsilon)  # Number of observations in the test set
ntr     <- n - nte

id.tr   <- sample(sample(sample(n)))[1:ntr]
id.te   <- setdiff(1: n, id.tr)

# 9 nearest neighbors
k        <- 9
y.te     <- y[id.te]                                 # True responses in test set
y.te.hat <- knn(x[id.tr,], x[id.te,], y[id.tr], k = k) # Predicted responses in test set

conf.mat.te <- table(y.te, y.te.hat)
conf.mat.te

# 1 nearest neighbor
k        <- 1
y.tr     <- y[id.tr]                                 # True responses in test set
y.tr.hat <- knn(x[id.tr,], x[id.tr,], y[id.tr], k = k) # Predicted responses in test set

conf.mat.tr <- table(y.tr, y.tr.hat)
conf.mat.tr

# ROC curve
library(ROCR)

y.roc <- y # as.factor(ifelse(y=='Yes',0,1))

kNN.mod <- class::knn(x[id.tr,], x[id.tr,], y.roc[id.tr], k = 3, prob = TRUE)
prob    <- attr(kNN.mod, 'prob')
prob    <- 2 * ifelse(kNN.mod == "0", 1-prob, prob) - 1

pred.knn <- prediction(prob, y.roc[id.tr])
perf.knn <- performance(pred.knn, measure = 'tpr', x.measure = 'fpr')

plot(perf.knn, col = 2, lwd = 2, lty = 2,
     main = paste('ROC curve for kNN with k=3'))
abline(a = 0, b = 1)

# Based on training set?
y.roc <- y #as.factor(ifelse(y=='Yes',0,1))

kNN.mod <- class::knn(x[id.tr,], x[id.tr,], y.roc[id.tr], k = 1, prob = TRUE)
prob    <- attr(kNN.mod, 'prob')
prob    <- 2*ifelse(kNN.mod == "0", 1-prob, prob) - 1

pred.1NN <- prediction(prob, y.roc[id.tr])
perf.1NN <- performance(pred.1NN, measure = 'tpr', x.measure = 'fpr')

kNN.mod <- class::knn(x[id.tr,], x[id.tr,], y.roc[id.tr], k = 13, prob = TRUE)
prob    <- attr(kNN.mod, 'prob')
prob    <- 2*ifelse(kNN.mod == "0", 1-prob, prob) - 1

pred.13NN <- prediction(prob, y.roc[id.tr])
perf.13NN <- performance(pred.13NN, measure = 'tpr', x.measure = 'fpr')

kNN.mod <- class::knn(x[id.tr,], x[id.tr,], y.roc[id.tr], k = 28, prob = TRUE)
prob    <- attr(kNN.mod, 'prob')
prob    <- 2*ifelse(kNN.mod == "0", 1-prob, prob) - 1

pred.28NN <- prediction(prob, y.roc[id.tr])
perf.28NN <- performance(pred.28NN, measure = 'tpr', x.measure = 'fpr')

plot(perf.1NN, col = 2, lwd = 2, lty = 2,
     main = paste('Comparative ROC curves in Training'))
plot(perf.13NN, col = 3, lwd = 2, lty = 3, add = TRUE)
plot(perf.28NN, col = 4, lwd = 2, lty = 4, add = TRUE)
abline(a = 0,b = 1)
legend('bottomright', inset = 0.05, c('1NN','13NN', '28NN'),
       col = 2:4, lty = 2:4)

# Test set
y.roc <- y #as.factor(ifelse(y=='Yes',0,1))

kNN.mod <- class::knn(x[id.tr,], x[id.te,], y.roc[id.tr], k = 1, prob = TRUE)
prob    <- attr(kNN.mod, 'prob')
prob    <- 2*ifelse(kNN.mod == "0", 1-prob, prob) - 1

pred.1NN <- prediction(prob, y.roc[id.te])
perf.1NN <- performance(pred.1NN, measure = 'tpr', x.measure = 'fpr')

kNN.mod <- class::knn(x[id.tr,], x[id.te,], y.roc[id.tr], k = 6, prob = TRUE)
prob    <- attr(kNN.mod, 'prob')
prob    <- 2*ifelse(kNN.mod == "0", 1-prob, prob) - 1

pred.6NN <- prediction(prob, y.roc[id.te])
perf.6NN <- performance(pred.6NN, measure = 'tpr', x.measure = 'fpr')

kNN.mod <- class::knn(x[id.tr,], x[id.te,], y.roc[id.tr], k = 13, prob = TRUE)
prob    <- attr(kNN.mod, 'prob')
prob    <- 2*ifelse(kNN.mod == "0", 1-prob, prob) - 1

pred.13NN <- prediction(prob, y.roc[id.te])
perf.13NN <- performance(pred.13NN, measure = 'tpr', x.measure = 'fpr')

kNN.mod <- class::knn(x[id.tr,], x[id.te,], y.roc[id.tr], k = 28, prob = TRUE)
prob    <- attr(kNN.mod, 'prob')
prob    <- 2*ifelse(kNN.mod == "0", 1-prob, prob) - 1

pred.28NN <- prediction(prob, y.roc[id.te])
perf.28NN <- performance(pred.28NN, measure = 'tpr', x.measure = 'fpr')

plot(perf.1NN, col = 2, lwd = 2, lty = 2,
     main = paste('Comparison of Predictive ROC curves'))
plot(perf.6NN, col = 3, lwd = 2, lty = 3, add = TRUE)
plot(perf.13NN, col = 4, lwd = 2, lty = 4, add = TRUE)
plot(perf.28NN, col = 5, lwd = 2, lty = 5, add = TRUE)
abline(a = 0,b = 1)
legend('bottomright', inset = 0.05, c('1NN','6NN','13NN', '28NN'),
       col = 2:5, lty = 2:5)