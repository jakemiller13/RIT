#
#    Collection of R functions written for
#       Teaching Stat 702 and Stat 846
#
#           (c) EPF, Spring 2010
#
#        source('functions-fokoue.R')
#

# Packages used by some functions

  library(MASS)

# Measures of goodness in binary classification

  measures <- function(label, response)
  {
     n<-length(label)   

     confmat     <- table(label, response)
     Accuracy    <- sum(diag(confmat))/n
     FPR         <- confmat[1,2]/rowSums(confmat)[1]
     TPR         <- confmat[2,2]/rowSums(confmat)[2] 
     FNR         <- confmat[2,1]/rowSums(confmat)[2]
     TNR         <- confmat[1,1]/rowSums(confmat)[1]
     Precision   <- confmat[2,2]/colSums(confmat)[2]
     Recall      <- TPR    
     Specificity <- TNR
     Sensitivity <- TPR
     F.measure   <- 2*(Precision*Recall)/(Precision+Recall)

     measured<-list(Accuracy=Accuracy, 
                    Precision = Precision,
                    F.measure = F.measure,
                    Recall=Recall,
                    FPR=FPR, TPR = TPR, 
                    FNR=FNR, TNR = TNR,
                    Specificity=Specificity,
                    Sensitivity=Sensitivity)

     return(measured)
  }


# Extract binary measures as vector for storage
  
  extract<-function(measure)
  {
     v<-numeric(10)
     v[1]<-measure$Accuracy
     v[2]<-measure$Precision
     v[3]<-measure$Recall
     v[4]<-measure$F.measure
     v[5]<-measure$FPR
     v[6]<-measure$TPR
     v[7]<-measure$FNR
     v[8]<-measure$TNR
     v[9]<-measure$Specificity
     v[10]<-measure$Sensitivity
     return(v)
  } 
    
# Generation of regression data
  
  generate.xy <- function(p=19, rho=0.81, tau=0.05, nn=199)
  { 
    # p is the dimensionality of the data 
    # rho is the correlation coefficient
    # tau is the overall level of relatedness
    # nn is the sample size

    pp     <- 1:p
    m1p    <- matrix(rep(pp,p), ncol=p, byrow=T)
    mp1    <- matrix(rep(pp,p), ncol=p, byrow=F)
    MmM    <- tau*abs(m1p-mp1)
    Sigmap <- rho^MmM
    mup    <- rep(0,p)
    mup    <- 1/(1:p)
    X      <- data.frame(mvrnorm(nn, mup, Sigmap))
    colnames(X) <- paste('X',1:p,sep='')
    Y      <- 1 + 2*X[,3] + X[,7] + 3*X[,9] + rnorm(nn) 
    return(data.frame(X,Y))  
  }

# Relative number of principal components

  intrinsic <- function(X, alpha)
  {   # X is  the matrix and 
      # alpha the percentage of variation desired
      v <- eigen(cov(X))$values
      pve <- cumsum(v/sum(v))
      return(min(which(pve>=alpha)))
  }  

# Perform PCA and extract the q important PC scores

  pca <- function(XY, q, XYnew)
  {
    p <- ncol(XY)-1
    pca.XY <- princomp(XY[,-(p+1)])
    Z  <- predict(pca.XY, XYnew[,-(p+1)])[,1:q]       
    ZY <- data.frame(Z, XYnew[,(p+1)])
    colnames(ZY) <-c(paste('Z',1:q,sep=''), 'Y') 
    return(ZY)
  } 
  
# Extract Principal Component Scores upto alpha

  pca.scores <- function(XY, alpha, XYnew)
  {
    p <- ncol(XY)-1
    pca.XY <- princomp(XY[,-(p+1)])
    q <- min(which(cumsum(pca.XY$sd/sum(pca.XY$sd))>=alpha))
    Z  <- predict(pca.XY, XYnew[,-(p+1)])[,1:q]       
    ZY <- data.frame(Z, XYnew[,(p+1)])
    colnames(ZY) <-c(paste('Z',1:q,sep=''), 'Y') 
    return(ZY)
  } 

# Adding Error bars for symmetrically distributed variables

  error.bar <- function(x, y, e)
  { 
     arrows(x,y+e, x, y-e, angle=90, code=3)
  }

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#     