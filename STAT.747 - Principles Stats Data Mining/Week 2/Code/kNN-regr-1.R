#
#    kNN - k Nearest Neighbors Regression Learning
#
#   Three weighting scheme and the Euclidean distance
#
#             source('kNN-regr-1.R')
#
#          Ernest Fokoue (c) Summer 2009
#

    graphics.off()
   
    source('input-scale-1.R')
    
    #xy   <- read.csv('cars-1.csv')
    
    xy    <- attitude
      
    colnames(xy)[1]<-'Y'
  
    y    <- xy[, 1]
    x    <- xy[,-1]
    x    <- cubitize(x) 
    n    <- nrow(xy)
 

#  Function for kNearest Neighbors 

   kNN <- function(x, y, xnew, k=1)
   {
      nnew     <- nrow(xnew)
      yhat.knn <- numeric(nnew)

      for(i in 1:nnew)
      {   
        xn <- matrix(rep(t(xnew[i,]), times=nrow(x)), byrow=T, ncol=ncol(x))
        dist.x <- sqrt(rowSums((as.matrix(xn)-as.matrix(x))^2))
        s.dist <- sort(dist.x, index=T)
        neighbors <- s.dist$ix[1:k]
        #w <- 1/k
        w <- exp(-dist.x[neighbors])/sum(exp(-dist.x[neighbors]))
        #w.x <- 1/(1+dist.x[neighbors]) 
        #w <- w.x/sum(w.x) 
        yhat.knn[i] <- sum(w*y[neighbors])        
      }
  
      return(list(yhat=yhat.knn)) 
   }  

#  Cross Validation (Leave One Out variety)

   K  <- n-1
   CV <- numeric(K)
 
   for(k in 1:K)
   { 
      yhat.cv <- numeric(n)
       
      for(i in 1:n)    
      {
         xy.cv   <- xy[-i, ]
         x.cv    <- xy.cv[,-1]
         y.cv    <- xy.cv[, 1]
         xnew    <- xy[i,-1]
         mod.knn <- kNN(x.cv, y.cv, xnew, k)
         yhat.cv[i] <- mod.knn$yhat
      } 
      CV[k] <- mean((y-yhat.cv)^2) 
   }

   k <- min(which(CV==min(CV)))
   
#  Explore all this now

   R      <- 100
   mse.te <- mse.tr <- matrix(0, nrow=R, ncol=2)

   for(r in 1:R)
   {
     
     id.tr <- sample(sample(n))[1:round(0.75*n)]
     xy.tr <- xy[ id.tr,]
     xy.te <- xy[-id.tr,] 

     knn.mod  <- kNN(xy.tr[,-1], xy.tr[,1], xy.tr[,-1], k=k)
     yhat.tr  <- knn.mod$yhat
     knn.mod  <- kNN(xy.tr[,-1], xy.tr[,1], xy.te[,-1], k=k)
     yhat.te  <- knn.mod$yhat

     mse.tr[r,1]  <- mean((xy.tr[,1]-yhat.tr)^2)    
     mse.te[r,1]  <- mean((xy.te[,1]-yhat.te)^2)
 
     lm.xy <- step(lm(Y~., data=xy.tr), ~., trace=0, direction='both')
     mse.tr[r,2]  <- mean((xy.tr[,1]-fitted(lm.xy))^2)
     mse.te[r,2]  <- mean((xy.te[,1]-predict(lm.xy, xy.te[,-1]))^2)    
    
   }

   x11()
   par(mfrow=c(1,2))
   boxplot(mse.tr)
   boxplot(mse.te)

   #cat('\n MSE(4NN) = ', mse.knn, '      MSE(LS) = ', mse.ls,'\n')    

   x11()
   plot(1:length(CV), CV, type='b', col='red', xlab='k', ylab='CV(k)',
       main='Cross Validation for finding k in kNN regression')

   print(t.test(mse.te[,1], mse.te[,2], paired=T))