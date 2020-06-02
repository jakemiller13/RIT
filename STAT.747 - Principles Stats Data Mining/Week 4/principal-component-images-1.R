#
#      Simple Demonstration of Principal Component Analysis of Images
#
#               source('principal-component-images-1.R)
#


#  
#
    
    library(ggplot2)
    library(proto)
    library(readr) 
    library(EBImage)

    #train.digit <- read.csv('train.csv')
    #labels      <- train.digit[, 1]
    #digits      <- train.digit[,-1] 
    #X <- as.matrix(digits[1:100,])
    #X <- read.csv('orl-faceshr.csv')
    #X <- as.matrix(digits[1:1000,]) 
    #W <- matrix(runif(p*k, 0,1), nrow=p, ncol=k)
    #H <- matrix(runif(k*n, 0,1), nrow=k, ncol=n)
    #n <- nrow(X)
    #p <- ncol(X)
    #k <- round(p/2)
    #X <- t(X)
    #p <- nrow(X)
    #n <- ncol(X)
    #k <- 64


    X <- read.csv('orl-faces2.csv')

    faces <- t(X)          # Rows are image attributes and columns are people   p x n

    r     <- 28
    c     <- 23
    p     <- nrow(faces)   # This is p = r x c
    n     <- ncol(faces)
    k     <- 64  
    pv    <- 0.80

    quartz()
    display(t(matrix(rowMeans(faces), nrow=28, ncol=23)), method = "raster", frame=0, all = TRUE)
    
    quartz()
    display(t(matrix(apply(faces,1,median), nrow=28, ncol=23)), method = "raster", frame=0, all = TRUE)
    
    mean.mat <- matrix(rep(rowMeans(faces), n), nrow=p, ncol=n)

    faces  <- faces - mean.mat

    pc.img <- princomp(faces)

    lambda  <- (pc.img$sdev)^2
    plambda <- cumsum(lambda/sum(lambda))
    k       <- min(which(plambda>=pv))
    
    cat(100*pv,'% of the variation gives us ', k, ' principal components\n')

    L      <- pc.img$loading
    Z      <- pc.img$scores 
    Wt     <- as.matrix(faces)%*%as.matrix(L)

    W      <- Wt[,1:k]

    feature.faces <- t(W)%*%faces

    quartz()
    par(mfrow=c(3,5))
    for(j in 1:15)
    {
       display(t(matrix(W[,j], nrow=28, ncol=23)), method = "raster", frame=0, all = TRUE)
    }

    quartz()
    par(mfrow=c(3,5))

    for(j in 1*(1:15))
    {
       display(t(matrix(faces[,j], nrow=28, ncol=23)), method = "raster", frame=0, all = TRUE)
    }


    selected <- sample(k, 3) 

    quartz()
    par(mfcol=c(2,3))

    for(j in selected)
    {
      display(t(matrix(faces[,j], nrow=28, ncol=23)), method = "raster", frame=0, all = TRUE)
      display(t(matrix(W[,j], nrow=28, ncol=23)), method = "raster", frame=0, all = TRUE)
    }

