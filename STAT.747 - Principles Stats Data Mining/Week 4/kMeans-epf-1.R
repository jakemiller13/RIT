##########################################################################
###      This is my simple quick R code for k-Means clustering           #
#           Here is the original dataset with all the labels             #
#        Not optimized, but good for understanding what k-Means does     #
#                                                                        #
#                  source('kMeans-clustering-1.R')                       #
#                                                                        #
#                    Spring 2010, Ernest Fokoue                          #
##########################################################################

# Clear off

  graphics.off()

# Libraries

  library(MASS)

# Here is the original dataset with all the labels
  
  n1<-165
  n2<-175
  n3<-190
  n<-n1+n2+n3
  k<-3

  x<-cbind(NULL)
  #x<-t(cbind(t(rnorm(n1,2,1)),t(rnorm(n2,9,1)), t(rnorm(n3,16,1))))
  library(mvtnorm)
  x1<-rmvnorm(n1, mean=c(7, 6), sigma=matrix(c(1, -.9, -.9, 2), nrow=2))
  x2<-rmvnorm(n2, mean=c(3, 9), sigma=matrix(c(1, .9, .9, 2), nrow=2))
  x3<-rmvnorm(n3, mean=c(0, 1), sigma=matrix(c(1, -.8, -.8, 2), nrow=2))
  x<-rbind(x1,x2,x3)

  Cs <- t(cbind(c(7, 6), c(3, 9), c(0, 1)))

  y<-t(cbind(t(rep(1,n1)), t(rep(2,n2)), t(rep(3,n3))))

  x0<-cbind(x,y)

  p<-dim(x)[2]
  n<-dim(x)[1]
  ind<-cbind(NULL)
  nb<- matrix(0, ncol=1, nrow=k)

  for(j in 1:(k-1))
  {
    nb[j]<-round(n/k)
    ind <- cbind(ind, t(rep(j, nb[j])))
  }
  
  nb[k]<- n-(k-1)*nb[k-1]
  ind <- t(cbind(ind, t(rep(k, nb[k]))))
  ind<-sample(sample(sample(sample(ind))))
  C<-oC<-matrix(0,nrow=k,ncol=p)
  x<-as.matrix(cbind(x,ind))
  ix<-x

  x11()
  par(mfrow=c(3,3))

  Sys.sleep(3)    # Pause between plot
  
  plot(x, col = x[,p+1])
  points(C, col = 6, pch = 15, cex=3)

  for(j in 1:k)
  {
    C[j, ]<-oC[j, ]<-colMeans(x[which(x[,p+1]==j),-(p+1)])
  }

  iC<-C

  it  <-0
  tol <- 0.000005
  iterate<-TRUE
  dist  <- matrix(0, nrow=n, ncol=k)
  while(iterate)
  {
    oC <- C
    it <- it + 1
    for(i in 1:n)
    {
      for(j in 1:k)
      {
        dist[i,j] <- sum((x[i,-(p+1)]-C[j,])^2)
      }
      x[i,p+1] <- which(dist[i,]==min(dist[i,]))
    }
   
    for(j in 1:k)
    {
      C[j, ]<-colMeans(x[which(x[,p+1]==j),-(p+1)])
    }
    
    Sys.sleep(3)    # Pause between plot
    
    plot(x, col = x[,p+1])
    points(C, col = 6, pch = 15, cex=3)
    iterate<-(min(sqrt(rowSums(oC-C)^2))>tol)
  }

  plot(x, col = y)
  points(Cs, col = 6, pch = 15, cex=3)
