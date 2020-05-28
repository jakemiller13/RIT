#
#          k Nearest Neighbors 
#    Cross Validation and Prediction
#      
#      source('knn-prediction-1.R')
#       (c) EPF, Spring 2009-2018
#

# Packages

  library(class)
  library(MASS)
  library(kernlab)
  library(mlbench)
  library(reshape2)
  library(ROCR)
  
# Clear

  graphics.off()
   
# Datasets are coming from the package kernlab

  data(PimaIndiansDiabetes)        # load the data
  xy <- PimaIndiansDiabetes        # Store data in xy frame
  help(PimaIndiansDiabetes)        # learn stuff about this dataset
  
  #xy <- xy[,-c(1,3)] # Throw away columns 1 and 3: clean a bit
   
  struct(xy)   # Check structure of data: here the response Y is in the last column
  
  n <- nrow(xy)       # Sample size
  p <- ncol(xy) - 1   # Dimensionality of the input space
  
  pos <- p+1          # Position of the response
  
  x   <- xy[,-pos]    # Data matrix: n x p matrix
  y   <- xy[, pos]    # Response vector
  
# Split the data
  
  set.seed (19671210) # Set seed for random number generation to be reproducible
  
  epsilon <- 1/3               # Proportion of observations in the test set
  nte     <- round(n*epsilon)  # Number of observations in the test set
  ntr     <- n - nte

  id.tr   <- sample(sample(sample(n)))[1:ntr]   # For a sample of ntr indices from {1,2,..,n}
  #id .tr <- sample(1:n, ntr, replace=F)        # Another way to draw from {1,2,..n}
  id.te   <- setdiff(1:n, id.tr)
  
  k        <- 9
  y.te     <- y[id.te]                                 # True responses in test set
  y.te.hat <- knn(x[id.tr,], x[id.te,], y[id.tr], k=k) # Predicted responses in test set
  
  k        <- 1
  y.tr     <- y[id.tr]                                 # True responses in test set
  y.tr.hat <- knn(x[id.tr,], x[id.tr,], y[id.tr], k=k) # Predicted responses in test set
  
  conf.mat.tr <- table(y.tr, y.tr.hat)
  
# Indicator of error in test
  
  ind.err.te <- ifelse(y.te!=y.te.hat,1,0)      # Random variable tracking error. Indicator

# Identification of misclassified cases
  
  id.err     <- which(y.te!=y.te.hat)           # Which obs are misclassified  
  
# Confusion matrix
  
  conf.mat.te <- table(y.te, y.te.hat)   
  
# Percentage of correctly classified (accurary)
  
  pcc.te <- sum(diag(conf.mat.te))/nte
  # pcc.te <- 1-mean(ind.err.te)        # Another way from ind.err.te
  # pcc.te <- 1-length(id.err.te)/nte   # Yet another way fr
  
# Test Error
  
  err.te <- 1-pcc.te                    # Complement of accurary

#
#  ROC Curve: Plotting one single ROC Curve
#
  library(ROCR)
  
  y.roc <- ifelse(y=='pos',1,0)
  
  kNN.mod <- class::knn(x[id.tr,], x[id.tr,], y.roc[id.tr], k=3, prob=TRUE)
  prob    <- attr(kNN.mod, 'prob')
  prob    <- 2*ifelse(kNN.mod == "0", 1-prob, prob) - 1
  
  pred.knn <- prediction(prob, y.roc[id.tr])
  perf.knn <- performance(pred.knn, measure='tpr', x.measure='fpr')
  
  plot(perf.knn, col=2, lwd= 2, lty=2, main=paste('ROC curve for kNN with k=3'))
  abline(a=0,b=1)
  
#
# Comparative ROC Curves on the training set
# 
  library(ROCR)
  
  y.roc <- ifelse(y=='pos',1,0)
  
  kNN.mod <- class::knn(x[id.tr,], x[id.tr,], y.roc[id.tr], k=1, prob=TRUE)
  prob    <- attr(kNN.mod, 'prob')
  prob    <- 2*ifelse(kNN.mod == "0", 1-prob, prob) - 1
  
  pred.1NN <- prediction(prob, y.roc[id.tr])
  perf.1NN <- performance(pred.1NN, measure='tpr', x.measure='fpr')
  
  kNN.mod <- class::knn(x[id.tr,], x[id.tr,], y.roc[id.tr], k=13, prob=TRUE)
  prob    <- attr(kNN.mod, 'prob')
  prob    <- 2*ifelse(kNN.mod == "0", 1-prob, prob) - 1
  
  pred.13NN <- prediction(prob, y.roc[id.tr])
  perf.13NN <- performance(pred.13NN, measure='tpr', x.measure='fpr')
  
  kNN.mod <- class::knn(x[id.tr,], x[id.tr,], y.roc[id.tr], k=28, prob=TRUE)
  prob    <- attr(kNN.mod, 'prob')
  prob    <- 2*ifelse(kNN.mod == "0", 1-prob, prob) - 1
  
  pred.28NN <- prediction(prob, y.roc[id.tr])
  perf.28NN <- performance(pred.28NN, measure='tpr', x.measure='fpr')
  
  plot(perf.1NN, col=2, lwd= 2, lty=2, main=paste('Comparative ROC curves in Training'))
  plot(perf.13NN, col=3, lwd= 2, lty=3, add=TRUE)
  plot(perf.28NN, col=4, lwd= 2, lty=4, add=TRUE)
  abline(a=0,b=1)
  legend('bottomright', inset=0.05, c('1NN','13NN', '28NN'),  col=2:4, lty=2:4)
  
#
# Comparative ROC Curves on the test set
# 
  library(ROCR)
  
  y.roc <- ifelse(y=='pos',1,0)
  
  kNN.mod <- class::knn(x[id.tr,], x[id.te,], y.roc[id.tr], k=1, prob=TRUE)
  prob    <- attr(kNN.mod, 'prob')
  prob    <- 2*ifelse(kNN.mod == "0", 1-prob, prob) - 1
  
  pred.1NN <- prediction(prob, y.roc[id.te])
  perf.1NN <- performance(pred.1NN, measure='tpr', x.measure='fpr')
  
  kNN.mod <- class::knn(x[id.tr,], x[id.te,], y.roc[id.tr], k=13, prob=TRUE)
  prob    <- attr(kNN.mod, 'prob')
  prob    <- 2*ifelse(kNN.mod == "0", 1-prob, prob) - 1
  
  pred.13NN <- prediction(prob, y.roc[id.te])
  perf.13NN <- performance(pred.13NN, measure='tpr', x.measure='fpr')
  
  kNN.mod <- class::knn(x[id.tr,], x[id.te,], y.roc[id.tr], k=28, prob=TRUE)
  prob    <- attr(kNN.mod, 'prob')
  prob    <- 2*ifelse(kNN.mod == "0", 1-prob, prob) - 1
  
  pred.28NN <- prediction(prob, y.roc[id.te])
  perf.28NN <- performance(pred.28NN, measure='tpr', x.measure='fpr')
  
  plot(perf.1NN, col=2, lwd= 2, lty=2, main=paste('Comparison of Predictive ROC curves'))
  plot(perf.13NN, col=3, lwd= 2, lty=3, add=TRUE)
  plot(perf.28NN, col=4, lwd= 2, lty=4, add=TRUE)
  abline(a=0,b=1)
  legend('bottomright', inset=0.05, c('1NN','13NN', '28NN'),  col=2:4, lty=2:4)
  
######################################################################  
#       Cross Validation for Tuning the neighborhood size k          #
######################################################################
  
  xtr <- x #x[id.tr,]          # Using the only the training for validation 
  ytr <- y #y[id.tr]           # We could use the entire sample but that's greedy
  
  vK <- seq(1, 25, by=1)    # Grid of values of k 
  nK <- length(vK)          # Number of values of k considered
  cv.error <-numeric(nK)    # Vector of cross validation errors for each k
  nc <- nrow(xtr)           # Number of observations used for cross validation 
  c   <- 10                 # Number of folds. We are doing c-fold cross validation
  
  S   <- sample(sample(nc)) # We randomly shuffle the data before starting CV
  m   <- ceiling(nc/c)      # Maximum Number of observations in each fold
  
  held.out.set <- matrix(0, nrow=c, ncol=m) # Table used to track the evolution
  
  for(ic in 1:(c-1))
  {
    held.out.set[ic,] <- S[((ic-1)*m + 1):(ic*m)]
  }
  held.out.set[c, 1:(nc-(c-1)*m)] <- S[((c-1)*m + 1):nc]  # Handling last chunk just in case n!=mc

#  
# Running the cross validation itself
#  
  for(j in 1:nK)
  { 
    for(i in 1:c)
    {   
      out <-  held.out.set[i,] 
      yhatc<- knn(xtr[-out,], xtr[out,],ytr[-out],  k=vK[j])
      cv.error[j]<-cv.error[j] + (length(out)-sum(diag(table(ytr[out],yhatc))))/length(out)
    }
    cv.error[j]<-cv.error[j]/c
  }

#  
# Plot the cross validation curve
#  
  plot(vK, cv.error, xlab='k', ylab=expression(CV[Error](k)), 
      main='Choice of k in k Nearest Neighbor by m-fold Cross Validation') 
  lines(vK, cv.error, type='c') 

#
#  Nicer plot with ggplot2
#
  
  cv <- data.frame(vK, cv.error)
  colnames(cv) <- c('k','error')
  
  ggplot(cv, aes(k,error))+geom_point()+geom_line()+
    labs(x='k=size of neighborhood', y=expression(CV[Error](k)))
  
# 
# Extract the optimal k yielded by cross validation 
#  
  k.opt.cv <- max(which(cv.error==min(cv.error)))
 
##############################################################
# Using the optimally tuned k let's estimate the test error  #
##############################################################
  
  set.seed (19671210)          # Set seed for random number generation to be reproducible
  
  epsilon <- 1/3               # Proportion of observations in the test set
  nte     <- round(n*epsilon)  # Number of observations in the test set
  ntr     <- n - nte
  
  k.opt.cv <- 28
  
  R <- 100   # Number of replications
  test.err <- numeric(R)
  
  for(r in 1:R)
  {
    # Split the data
    
    id.tr   <- sample(sample(sample(n)))[1:ntr]                   # For a sample of ntr indices from {1,2,..,n}
    id.te   <- setdiff(1:n, id.tr)
  
    y.te         <- y[id.te]                                        # True responses in test set
    y.te.hat     <- knn(x[id.tr,], x[id.te,], y[id.tr], k=k.opt.cv) # Predicted responses in test set
    ind.err.te   <- ifelse(y.te!=y.te.hat,1,0)                      # Random variable tracking error. Indicator
    test.err[r]  <- mean(ind.err.te)
  }  
  
  test <- data.frame(test.err)
  colnames(test) <- c('error')
  
  ggplot(test, aes(x='', y=error))+geom_boxplot()+
  labs(x='Method', y=expression(hat(R)[te](kNN)))
  
  ##############################################################
  #  Predictively compare different kNN  learning machines     #
  ##############################################################
  
  set.seed (19671210)          # Set seed for random number generation to be reproducible
  
  epsilon <- 1/3               # Proportion of observations in the test set
  nte     <- round(n*epsilon)  # Number of observations in the test set
  ntr     <- n - nte
  
  R <- 100   # Number of replications
  test.err <- matrix(0, nrow=R, ncol=3)
  
  for(r in 1:R)
  {
    # Split the data
    
    id.tr   <- sample(sample(sample(n)))[1:ntr]                   # For a sample of ntr indices from {1,2,..,n}
    id.te   <- setdiff(1:n, id.tr)
    
    y.te         <- y[id.te]                                        # True responses in test set
    
    # First machine: 1NN
    
    y.te.hat     <- knn(x[id.tr,], x[id.te,], y[id.tr], k=1)        # Predicted responses in test set
    ind.err.te   <- ifelse(y.te!=y.te.hat,1,0)                      # Random variable tracking error. Indicator
    test.err[r,1]  <- mean(ind.err.te)
    
    # Second machine: Our optimal NN found earlier with k=k.opt.cv
    y.te.hat     <- knn(x[id.tr,], x[id.te,], y[id.tr], k=k.opt.cv) # Predicted responses in test set
    ind.err.te   <- ifelse(y.te!=y.te.hat,1,0)                      # Random variable tracking error. Indicator
    test.err[r,2]  <- mean(ind.err.te)
    
    # Third machine: k=round(sqrt(n))
    y.te.hat     <- knn(x[id.tr,], x[id.te,], y[id.tr], k=round(sqrt(n)))       # Predicted responses in test set
    ind.err.te   <- ifelse(y.te!=y.te.hat,1,0)                      # Random variable tracking error. Indicator
    test.err[r,3]  <- mean(ind.err.te)
    
  }  
  
  test <- data.frame(test.err)
  Method<-c('1NN', 'k.opt.NN', 'k.r.NN')
  colnames(test) <- Method
  boxplot(test)
  
  
  require(reshape2)
  ggplot(data = melt(test), aes(x=variable, y=value)) + geom_boxplot(aes(fill=variable))+
    labs(x='Method', y=expression(hat(R)[te](kNN)))+
    theme(legend.position="none") 

#  
# Is the difference between the methods significant
#  
  aov.method <- aov(value~variable, data=melt(test))
  anova(aov.method)
  #summary(aov.method)
  
  TukeyHSD(aov.method, ordered = TRUE)
  plot(TukeyHSD(aov.method))
  