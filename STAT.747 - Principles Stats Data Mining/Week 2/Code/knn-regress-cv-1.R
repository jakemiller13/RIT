#
#       k Nearest Neighbors 
#   Cross Validation and Weighting
#      
#      source('knn-classif-cv-1.R')
#       (c) EPF, Spring 2010
#

# Packages

  library(class)
  library(FNN)
  library(MASS)

# Clear

  graphics.off()
   
# Dataset

  #XY.tr <- read.csv('german-credit.csv', header=T)
  #XY.tr <- read.csv('breastcancer.csv', header=T)
  #XY.tr <- cbind(crabs[,-c(1,2,3)], crabs[,2])
  #XY.tr <- Pima.tr
  
  #XY.tr <- as.matrix(attitude[,ncol(attitude):1])
  
  XY.tr <- read.csv('boston.csv')
  
  p     <- ncol(XY.tr)-1   
  X.tr  <- XY.tr[,-(p+1)]
  Y.tr  <- XY.tr[, (p+1)]
  ntr   <- nrow(XY.tr)
 
# Cross Validation for Tuning

  vC <- seq(1, 15, by=1)
  nC <- length(vC)
  error <-numeric(nC)
  
  nc <- ntr

  k   <- 1 # round(nc/10)
  S   <- sample(sample(nc))
  c   <- ceiling(nc/k)
  held.out.set <- matrix(0, nrow=c, ncol=k) 

  for(ic in 1:(c-1)){held.out.set[ic,] <- S[((ic-1)*k + 1):(ic*k)]}
  held.out.set[c, 1:(nc-(c-1)*k)] <- S[((c-1)*k + 1):nc]
 
  for(j in 1:nC)
  { 
    for(i in 1:c)
    {   
      out       <- held.out.set[i,] 
      yhatc     <- knn.reg(X.tr[-out,], X.tr[out,],Y.tr[-out],  k=vC[j])$pred
      y         <- Y.tr[out]
      error[j]  <- error[j] + mean((y-yhatc)^2)   
    }
    error[j]  <- error[j]/c 
  }

  x11() 
  plot(vC, error, xlab='k', ylab='CV(k)', 
      main='Choice of k in kNN by Cross Validation') 
  lines(vC, error, type='c') 
 
 