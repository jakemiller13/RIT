#
#   Demonstration of the benefits of kernels
#     Artificial and Real life examples
#
#       (c) Ernest Fokoue, Spring 2010
#
#            source('doughnuts.R')


library(MASS)
library(kernlab)
library(scatterplot3d)

graphics.off()

n <- 400   # Sample size
m <- 75    # Number of replications
homogeneous   <- matrix(0, ncol=5, nrow=m)
heterogeneous <- matrix(0, ncol=5, nrow=m)

#
#                      Part I
#
#   Observations are homogeneous within their class
#   Despite apparent difficulty in 2-D QDA does wonders
#     Because there is only one matrix per lass
# The differences in performance are not substantial
#
#

# Run a total of m replications and compute error each time

  for(r in 1:m)
  {
    x <-  mvrnorm(n, c(0,0),matrix(c(3,0,0,3),2,2))
    y <- (x[,1]^2+x[,2]^2)<=6
    y <- as.numeric(y)
    y <- factor(y)

    conf.homogene.raw<-table(predict(qda(x,y), x)$class, y)
    accuracy.homogene.raw <- sum(diag(conf.homogene.raw))/n
    homogeneous[r, 1] <- accuracy.homogene.raw    

    z1<- x[,1]^2
    z2<- sqrt(2)*x[,1]*x[,2]
    z3<- x[,2]^2
    Z<- data.frame(z1,z2,z3)

    if (r == 1)
    {     
      x11()
      plot(x,col=y, xlab='x1', ylab='x2', pch=LETTERS[y],
      main='Original Homogeneous 2-dimensional Data')
      x11()
      scatterplot3d(z1,z2,z3, pch=LETTERS[y],
      main='Homogeneous Data Projected onto 3-dimensional space')
    }
    
    # Quadractic discrimination on homogeneous projected 

    conf.homogene.projected<-table(predict(qda(Z,y), Z)$class, y)
    accuracy.homogene.projected <- sum(diag(conf.homogene.projected))/n
    homogeneous[r, 2] <- accuracy.homogene.projected
    
    # Support Vector Machines

    conf.homogene.svm<-table(predict(ksvm(y~x, kernel='rbfdot'), x), y)
    accuracy.homogene.svm <- sum(diag(conf.homogene.svm))/n
    homogeneous[r, 3] <- accuracy.homogene.svm

    # Logistic regression

    xy<-data.frame(y,x)
    conf.homogene.logistic<-table(predict(glm(y~., data=xy,family='binomial'), 
    data.frame(xy[,-1]), type='response')>0.5, xy[,1])
    accuracy.homogene.logistic <- sum(diag(conf.homogene.logistic))/n
    homogeneous[r, 4] <- accuracy.homogene.logistic
  
    # Logistic regression projected

    xy<-data.frame(y,Z)
    conf.homogene.logistic.projected<-table(predict(glm(y~., data=xy,family='binomial'), 
    data.frame(xy[,-1]), type='response')>0.5, xy[,1])
    accuracy.homogene.logistic.projected <- sum(diag(conf.homogene.logistic.projected))/n
    homogeneous[r, 5] <- accuracy.homogene.logistic.projected

# Save this homogeneous data

  XY <- data.frame(x,y)
  write.table(XY, file='doughnuts-easy.csv', sep=',', row.names=F)   

#
#                      Part II
#
#   Observations are nonhomogeneous within their class
#   QDA cannot do well anymore, Kernels are badly needed
#           Support Vector Machine Wins Big
#

# Mess up the second class with guys who used to belong to the first


    rm(y)          # Remove the previous class labels

    center <- (x[,1]^2+x[,2]^2)<=1 
    y <- (x[,1]^2+x[,2]^2)<=6
    y <- as.numeric(y)
    y[center] <- 0
    y <- factor(y)

    conf.heterogene.raw<-table(predict(qda(x,y), x)$class, y)
    accuracy.heterogene.raw <- sum(diag(conf.heterogene.raw))/n
    heterogeneous[r, 1] <- accuracy.heterogene.raw    

    z1<- x[,1]^2
    z2<- sqrt(2)*x[,1]*x[,2]
    z3<- x[,2]^2
    Z<- data.frame(z1,z2,z3)

    if (r == 1)
    {     
      x11()
      plot(x,col=y, xlab='x1', ylab='x2', pch=LETTERS[y],
      main='Original Heterogeneous 2-dimensional Data')
      x11()
      scatterplot3d(z1,z2,z3, pch=LETTERS[y],
      main='Heterogeneous Data Projected onto 3-dimensional space')
    }
    
    # Quadractic discrimination on homogeneous projected 

    conf.heterogene.projected<-table(predict(qda(Z,y), Z)$class, y)
    accuracy.heterogene.projected <- sum(diag(conf.heterogene.projected))/n
    heterogeneous[r, 2] <- accuracy.heterogene.projected

       
    # Support Vector Machines

    conf.heterogene.svm<-table(predict(ksvm(y~x, kernel='rbfdot'), x), y)
    accuracy.heterogene.svm <- sum(diag(conf.heterogene.svm))/n
    heterogeneous[r, 3] <- accuracy.heterogene.svm

    # Logistic regression

    xy<-data.frame(y,x)
    conf.heterogene.logistic<-table(predict(glm(y~., data=xy,family='binomial'), 
    data.frame(xy[,-1]), type='response')>0.5, xy[,1])
    accuracy.heterogene.logistic <- sum(diag(conf.heterogene.logistic))/n
    heterogeneous[r, 4] <- accuracy.heterogene.logistic

    # Logistic regression projected

    xy<-data.frame(y,Z)
    conf.heterogene.logistic.projected<-table(predict(glm(y~., data=xy,family='binomial'), 
    data.frame(xy[,-1]), type='response')>0.5, xy[,1])
    accuracy.heterogene.logistic.projected <- sum(diag(conf.heterogene.logistic.projected))/n
    heterogeneous[r, 5] <- accuracy.heterogene.logistic.projected    

  }

# Let's summarize our findings and display them
  
  cat(rep('\n',126))
  dough<-rbind(colMeans(homogeneous), colMeans(heterogeneous))
  dough<- data.frame(dough)
  rownames(dough)<- c('Homogeneous','Heterogeneous')
  colnames(dough)<- c('QDA-raw', 'QDA-projected',  'SVM-Raw', 'Logistic-Raw','Logistic-Projected')
  print(dough)

# 
  XY <- data.frame(x,y)
  write.table(XY, file='doughnuts.csv', sep=',', row.names=F)   