#
#  Predictive Analytics with k-Nearest Neighbors   
#
#          source('kNN-predictive-1.R')
#
#              Winter 2013, EPF
#


# Unitizing the data

  unitlength <-function(xx)
  {
    n <- nrow(xx)
    p <- ncol(xx)
    aa  <- matrix(rep(apply(xx,2,mean), n), ncol=p, byrow=TRUE)
    bb  <- sqrt((n-1)*matrix(rep(apply(xx,2,var), n), ncol=p, byrow=TRUE))
    return((xx-aa)/bb)
  }

# Standardizing the data

  standard <-function(xx)
  {
    n <- nrow(xx)
    p <- ncol(xx)
    aa  <- matrix(rep(apply(xx,2,mean), n), ncol=p, byrow=TRUE)
    bb  <- sqrt(matrix(rep(apply(xx,2,var), n), ncol=p, byrow=TRUE))
    return((xx-aa)/bb)
  }


# Cubifying the data

  unitcube <-function(xx)
  {
    n <- nrow(xx)
    p <- ncol(xx)
    aa  <- matrix(rep(apply(xx,2,min), n), ncol=p, byrow=TRUE)
    bb  <- matrix(rep(apply(xx,2,max), n), ncol=p, byrow=TRUE)    
	return((xx-aa)/(bb-aa))
  }

#  Clear screen
   
   graphics.off()

#  Set the stage

   library(caret)
   library(FNN)
   library(kernlab)
   
   library(car)
   
   kern = 'rbfdot'
  
   set.seed(19671210)
   
   pos <- 2
  
   #xy[,-pos] <- standard(xy[,-pos])
   #xy[,-pos] <- unitlength(xy[,-pos])
   #xy[,-pos]  <- unitcube(xy[,-pos])
   
   n <- 40

   x <- seq(-2,2, length=n)
   f <- function(x){return(1+x^3-2*x+x^2)}
   sigma <- 1
   
   R <- 100
   K <- 15
   mse.knn <- matrix(0, ncol=K, nrow=R)
   mse.rvm <- mse.svm <- mse.lm <- kNN.k <- numeric(R)

   for(r in 1:R)
   {
      noise <- rnorm(n, mean=0, sd=sigma)
      y     <- f(x) + noise
      xy    <- data.frame(x,y)
      
      ntr   <- round((2/3)*n)
      nte   <- n - ntr
      id.tr <- sample(sample(sample(n)))[1:ntr]
      id.te <- (1:n)[-id.tr]

      xy.tr <- xy[id.tr, ]
      xy.te <- xy[id.te, ]
  
      for(k in 1:K)
      {
        fit.te.knn   <- knn.reg(train=data.frame(xy.tr[,-pos]),test=data.frame(xy.te[,-pos]), y=xy.tr[,pos], k=k)   
        yhat.te.knn  <- fit.te.knn$pred
        mse.knn[r,k] <- mean((xy.te[,pos] - yhat.te.knn)^2)      
      }
    
  	kNN.k[r]       <- which(mse.knn[r,]==min(mse.knn[r,]))
 	
    svm.xy         <- ksvm(y~., data=xy.tr, kernel=kern)
	  rvm.xy         <- gausspr(y~., data=xy.tr, kernel=kern)
	  lm.xy          <- lm(y~., data=xy.tr)
	  
	  yhat.te.rvm    <- predict(rvm.xy, xy.te[,-pos])
 	  yhat.te.svm    <- predict(svm.xy, xy.te[,-pos])
    yhat.te.lm     <- predict(lm.xy, x=xy.te[,-pos])
	  
    mse.rvm[r]     <- mean((xy.te[,pos] - yhat.te.rvm)^2)
	  mse.svm[r]     <- mean((xy.te[,pos] - yhat.te.svm)^2)
    mse.lm[r]      <- mean((xy.te[,pos] - yhat.te.lm)^2)
    
   }

   x11()
   par(mfrow=c(1,1))
   plot(1:K, mse.knn[R,], xlab ='Number of neighbors', ylab='MSE(Test)', type='b')    


   nk <- length(unique(kNN.k))
   freq.k <- numeric(nk)
   for(j in 1:nk)
   {
	  freq.k[j] <- length(which(kNN.k == unique(kNN.k)[j]))
   }
    
   k.opt <- unique(kNN.k)[which(freq.k==max(freq.k))]

   x11()
   yhat.knn <- knn.reg(train=data.frame(xy[,1]),data.frame(test=xy[,1]), y=xy[,2], k=k.opt[1])$pred
   plot(x,y, col=1, lty =0, type='p', pch=1)
   lines(x,yhat.knn, type='l', lwd= 2, lty =2, col=2, pch='')
   lines(x,f(x), type='l', lwd= 3, lty =3, col=3, pch='')
   legend('top', inset=0.02, c('Noisy data points', 'kNearest Neighbors Fit', 'True function'),
          lty =c(0,2,3), col=c(1,2,3), pch=c(1,NA,NA), lwd=c(0,2,3))
   
   x11()
   boxplot(kNN.k, horizontal=F, names=c('Number of neighbors'))
    
   x11()
   hist(kNN.k, xlab='Number of neighbors', ylab='Frequency')

   x11()
   boxplot(mse.lm, mse.knn[,k.opt], mse.rvm, mse.svm, horizontal=F, names=c('OLS', 'kNN', 'RVM', 'SVM'), ylab='Empirical Mean Squared Error')
