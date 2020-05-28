library(class)
library(MASS)
library(kernlab)
#library(mlbench)
library(reshape2)
library(ROCR)
library(ggplot2)
library(ada)
library(adabag)
library(ipred)
library(survival)
library(rchallenge)
library(PerformanceAnalytics)
library(knitr)
library(acepack)
library(caret)
library(HSAUR2)
library(corrplot)

# Datasets
d = '/Users/Jake/Google Drive/RIT/STAT.747 - Principles Stats Data Mining/Data'
xy <- read.csv(paste(d, '/banana-shaped-data-1.csv', sep = ''))
xy <- read.csv(paste(d, '/doughnuts-easy.csv', sep = ''))
xy <- read.csv(paste(d, '/doughnuts.csv', sep = ''))
xy <- read.csv(paste(d, '/four-corners-data-1.csv', sep = ''))
xy <- read.csv(paste(d, '/simple-2d-for-knn-2.csv', sep = ''));
  colnames(xy)[2:3]<-c('X1', 'X2')
xy <- read.csv(paste(d, '/simple-2d-for-knn.csv', sep = ''));
  colnames(xy)[2:3]<-c('X1', 'X2')
#xy <- read.csv(paste(d, '/class-faithful.csv', sep = ''));
#  colnames(xy) <- c('X1', 'X2', 'y'); xy[,3] <- ifelse(xy[,3] == 1, 1, 0)

# Check first/last 6 observations
head(xy)
tail(xy)

# Latex table - not displaying great
library(xtable)
xtable(head(xy))

# Types of variables
str(xy)

# Constants
n <- nrow(xy)
p <- ncol(xy) - 1
pos <- 1
x <- xy[,-pos]
y <- as.factor(xy[, pos])
n; p

# Correlation
library(corrplot)
corr.x <- cor(xy[,-pos])
corrplot(corr.x)

# EDA
plot(xy[,-pos], col=xy[,pos] + 2)

# KernSmooth
library(KernSmooth)

x <- xy[,-pos]
dens.est <- bkde2D(x, bandwidth = sapply(x, dpik))
plot(x, xlab = expression(X[1]), ylab = expression(X[2]))

# Contour plot
contour(x = dens.est$x1, y = dens.est$x2, z = dens.est$fhat, add = TRUE)

# Logistic regression
logistic.xy <- glm(as.factor(y) ~ ., data = xy,
                   family = binomial(link = 'logit'))
summary(logistic.xy)

aic.xy <- step(logistic.xy, scope = ~., direction = 'both',
               k = 2, trace = 0)
summary(aic.xy)

bic.xy <- step(logistic.xy, scope = ~., direction = 'both',
               k = log(n), trace = 0)
summary(bic.xy)

# Histogram/boxplot
par(mfrow = c(1,2))
hist(xy$X1, prob = T)
lines(density(xy$X1), lwd = 2, col = 'red')
boxplot(xy$X1)

# Boxplots
par(mfrow = c(1,2))
boxplot(X1 ~ y, data = xy, col = 2:3)
boxplot(X2 ~ y, data = xy, col = 2:3)

log.acc <- glm(y ~ X1, data = xy, family = binomial(link = 'logit'))
summary(log.acc)

# Train/test set
set.seed(19671210)

epsilon <- 1/3
nte <- round(n * epsilon)
ntr <- n - nte

id.tr <- sample(sample(sample(n)))[1: ntr]
id.te <- setdiff(1:n, id.tr)

# Stratified holdout subsampling
stratified.holdout <- function(y, ptr)
{
  n         <- length(y)
  labels    <- unique(y)
  id.tr <- id.te <- NULL
  
  y <- sample(sample(sample(y)))
  
  for(j in 1:length(labels))
  {
    sj <- which(y == labels[j])
    nj <- length(sj)
    
    id.tr <- c(id.tr, (sample(sample(sample(sj))))[1: round(nj * ptr)])
  }
  
  id.te <- (1:n)[-id.tr]
  
  return(list(idx1 = id.tr, idx2 = id.te))
}

hold <- stratified.holdout(as.factor(xy[,pos]), 1 - epsilon)
id.tr <- hold$idx1
id.te <- hold$idx2
ntr <- length(id.tr)
nte <- length(id.te)

# kNN with k = 9
library(class)

k = 9
y.te <- y[id.te]
y.te.hat <- knn(x[id.tr,], x[id.te,], y[id.tr], k = k)

conf.mat.te <-table(y.te, y.te.hat)
conf.mat.te

# kNN with k = 1
k <- 1
y.tr <- y[id.tr]
y.tr.hat <- knn(x[id.tr,], x[id.tr,], y[id.tr], k = k)

conf.mat.tr <- table(y.tr, y.tr.hat)
conf.mat.tr

# ROC Curves
library(ROCR)

y.roc <- as.factor(y)

kNN.mod <- class::knn(x[id.tr,], x[id.tr,], y.roc[id.tr], k = 3, prob = TRUE)
prob <- attr(kNN.mod, 'prob')
prob <- 2*ifelse(kNN.mod == '0', 1 - prob, prob) - 1

pred.knn <- prediction(prob, y.roc[id.tr])
perf.knn <- performance(pred.knn, measure = 'tpr', x.measure = 'fpr')

plot(perf.knn, col = 2, lwd = 2, lty = 2,
     main = paste('ROC curve for kNN with k = 3'))
abline(a = 0, b = 1)

# ROC curve based on training set

y.roc <- as.factor(y)

kNN.mod <- class::knn(x[id.tr,], x[id.tr,], y.roc[id.tr], k = 1, prob = TRUE)
prob <- attr(kNN.mod, 'prob')
prob <- 2*ifelse(kNN.mod == '0', 1 - prob, prob) - 1

pred.1NN <- prediction(prob, y.roc[id.tr])
perf.1NN <- performance(pred.1NN, measure = 'tpr', x.measure = 'fpr')

kNN.mod <- class::knn(x[id.tr,], x[id.tr,], y.roc[id.tr], k = 13, prob = TRUE)
prob <- attr(kNN.mod, 'prob')
prob <- 2*ifelse(kNN.mod == '0', 1 - prob, prob) - 1

pred.13NN <- prediction(prob, y.roc[id.tr])
perf.13NN <- performance(pred.13NN, measure = 'tpr', x.measure = 'fpr')

kNN.mod <- class::knn(x[id.tr,], x[id.tr,], y.roc[id.tr], k = 28, prob = TRUE)
prob <- attr(kNN.mod, 'prob')
prob <- 2*ifelse(kNN.mod == '0', 1 - prob, prob) - 1

pred.28NN <- prediction(prob, y.roc[id.tr])
perf.28NN <- performance(pred.28NN, measure = 'tpr', x.measure = 'fpr')

plot(perf.1NN, col = 2, lwd = 2, lty = 2,
     main = paste('Comparative ROC curves in training'))
plot(perf.13NN, col = 3, lwd = 2, lty = 3, add = TRUE)
plot(perf.28NN, col = 4, lwd = 2, lty = 4, add = TRUE)
abline(a = 0, b = 1)
legend('bottomright', inset = 0.05, c('1NN', '13NN', '28NN'),
       col = 2:4, lty = 2:4)

# ROC curves based on test set
y.roc <- as.factor(y)

kNN.mod <- class::knn(x[id.tr,], x[id.te,], y.roc[id.tr], k = 1, prob = TRUE)
prob <- attr(kNN.mod, 'prob')
prob <- 2*ifelse(kNN.mod == '0', 1 - prob, prob) - 1

pred.1NN <- prediction(prob, y.roc[id.te])
perf.1NN <- performance(pred.1NN, measure = 'tpr', x.measure = 'fpr')

kNN.mod <- class::knn(x[id.tr,], x[id.te,], y.roc[id.tr], k = 13, prob = TRUE)
prob <- attr(kNN.mod, 'prob')
prob <- 2*ifelse(kNN.mod == '0', 1 - prob, prob) - 1

pred.13NN <- prediction(prob, y.roc[id.te])
perf.13NN <- performance(pred.13NN, measure = 'tpr', x.measure = 'fpr')

kNN.mod <- class::knn(x[id.tr,], x[id.te,], y.roc[id.tr], k = 28, prob = TRUE)
prob <- attr(kNN.mod, 'prob')
prob <- 2*ifelse(kNN.mod == '0', 1 - prob, prob) - 1

pred.28NN <- prediction(prob, y.roc[id.te])
perf.28NN <- performance(pred.28NN, measure = 'tpr', x.measure = 'fpr')

plot(perf.1NN, col = 2, lwd = 2, lty = 2,
     main = paste('Comparative ROC curves in training'))
plot(perf.13NN, col = 3, lwd = 2, lty = 3, add = TRUE)
plot(perf.28NN, col = 4, lwd = 2, lty = 4, add = TRUE)
abline(a = 0, b = 1)
legend('bottomright', inset = 0.05, c('1NN', '13NN', '28NN'),
       col = 2:4, lty = 2:4)