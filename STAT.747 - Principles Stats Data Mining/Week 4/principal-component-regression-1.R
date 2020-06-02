#
#        Principal  Component  Regression
#          Simple and Complex Examples
#
#            (c) EPF, Spring 2010
#
#   source('principal-component-regression-1.R')
#


# Packages needed

  library(MASS)

# Clear

  graphics.off()

# PRESS statistic function

  press <- function(mod){return(sum((resid(mod)/(1-hatvalues(mod)))^2))}
  
# Load the data

  
  
  
  
  library(ridge)
  data(Hald)       
  XY <- as.data.frame(Hald)   # Response at 1
  
   XY <- longley
  
# Transform the data

  p   <- ncol(XY)-1
  n   <- nrow(XY)
  pos <- 1
  colnames(XY)[pos] <- 'Y'

  X <- XY[,-pos]
  Y <- XY[, pos]
  
# Regression on original variables

  lm.xy <- lm(Y~., data=XY)
  lm.xy <- step(lm.xy, Y~., direction = "both", k=log(n),trace=0)

  R.squared.original <- summary(lm.xy)$r.squared

# Perform PCA

  pca.xy <- princomp(XY[,-pos])
  Z <- pca.xy$scores
  Z1 <- predict(pca.xy,XY[,-pos]) # same as scores
  

# Effect of number of components
  
  k     <- p
  v.rsq <- numeric(k-1)
 
  for(q in 2:k)
  {
     ZY <- data.frame(Z[,1:q], XY[,pos])
     colnames(ZY)[q+1] <- 'Y'  
     lm.zy <- lm(Y~., data=ZY)
     v.rsq[q-1]<- summary(lm.zy)$r.squared
  }

  plot(2:k, v.rsq, type='l', col='red', xlab = 'number of components', ylab='R^2',
       main='R squared as a function of number of components')

  x11()
  par(mfrow=c(2,2))
  plot(lm.zy)

  # Check the residuals
  # Investigate the meaning of outliers

  lm.xy <- step(lm(Y~., data=XY), Y~., direction='both', k=log(nn), trace=0)
  print(summary(lm.xy))

  x11()
  par(mfrow=c(2,2))
  plot(lm.xy, main='Regression on Original Variables') 

  #x11()
  #plot(X, main='Scatterplot of X')

  pca.x <- princomp(X)
  Z     <- predict(pca.x,X)
  BIC   <- PRESS <- numeric(p)

  for(q in 1:p)
  {
    ZY       <- data.frame(Z[,1:q],Y)
    lm.zy    <- lm(Y~., data=ZY)
    BIC[q] <- AIC(lm.zy, k=log(nn))
	PRESS[q] <- press(lm.zy)
  }

  x11()
  par(mfrow=c(2,2))
  plot(lm.zy, main='Regression on Principal Component Scores')

   
  x11()
  plot(eigen(cov(X))$values, type='l', col='red', xlab='index', 
       ylab='eigenvalues', main='Screeplot')

  x11()
  plot(1:p, BIC, type='l', col='red', xlab='# PCs', 
       ylab='BIC', main='Effect of number of PC on BIC')
	   
  x11()
  plot(1:p, PRESS, type='l', col='red', xlab='# PCs', 
       ylab='PRESS', main='Effect of number of PC on the PRESS statistic')	   
	   
	   

 