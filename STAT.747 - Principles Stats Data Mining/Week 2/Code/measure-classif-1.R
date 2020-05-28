#
#         Exploration of Classification
#    Various Measures of Performance computed
#
#        (c) Ernest Fokoue, Spring 2010
#
#         source('measure-classif-1.R')
#        


# Clear all
  
  graphics.off()
  #rm(list=ls())

# Load the packages and that data

  library(MASS)
  library(kernlab) 

# What the task under consideration

   # txtdata <- c('Doughnuts Data')
   # txtdata <- c('Wisconsin Breast Cancer Data')
   # txtdata <- c('Pima Indian Diabetes Data')
     txtdata <- c('German Credit Data')
   # txtdata <- c('eMail Spam Data')
   # txtdata <- c('Mines and Rocks')
   # txtdata <- c('Crabs Leptograpsus') 
   # txtdata <- c('Liver Disorders BUPA[UK]')
 
# Prepare this data for nice binary classfication
  
  #XY   <- read.csv('doughnuts.csv', header=T)
  #XY   <- read.csv('breast-cancer.csv', header=T)
  #XY   <- read.csv('Pima.csv', header=T)
   XY   <- read.csv('german-credit.csv', header=T)
  #XY   <- read.csv('spam.csv', header=T)
  #XY   <- read.csv('minesNrocks.csv', header=T)
  #XY   <- read.csv('crabs-sex.csv', header=T)
  #XY   <- read.csv('liver.csv', header=T)
 
  n    <- nrow(XY)
  p    <- ncol(XY)-1  
  colnames(XY)[p+1]<-'Y'

  response <- rep(0, n)
  positive <- which(XY$Y==1) 
  response[positive] <- 1
  response <- factor(response)
  XY$Y <- response
  XY <- data.frame(XY)
  cutoff <- 0.5

# Have a look at the data
  
  ind.bp <- sample(1:p,6 , replace=T)
  par(mfrow=c(2,3))
  for(i in 1:6)
  {
    j <- ind.bp[i]
    boxplot(XY[,j]~XY[,p+1], col=(j+1):(j+2),
    main=paste('Boxplot of ',colnames(XY)[j]), 
    names=c('Negative','Positive'))
  } 

# Fit logistic regression and perform stepwise model selection
  
  logistic.XY   <- glm(Y~., family= binomial, data=XY)
  logistic.XY.f <- step(logistic.XY, Y~., direction="both", trace=0,
                          k=log(n))  
 
  Predicted <- rep(0, nrow(XY))
  Predicted[which(predict(logistic.XY.f, XY[,-(p+1)], 
           type = "response") > cutoff)]<-1
  Predicted <- factor(Predicted)

# Measuring the  performance of the classifiers

  label    <- XY$Y
  response <- Predicted
  
# Confusion matrix
  
  print(table(label,response))

  lr.p     <- measures(label, response)
  meas1    <- round(c(lr.p$Accuracy, lr.p$FPR, 
              lr.p$TPR, lr.p$FNR, lr.p$TNR),2)
  meas1a   <- round(c(lr.p$F.measure, lr.p$Precision, lr.p$Recall, 
              lr.p$Specificity, lr.p$Sensitivity),2)

  pred.ld  <- predict(lda(XY[,-(p+1)], XY[,p+1]),XY[,-(p+1)])$class
  ld.p     <- measures(label, pred.ld)
  print(table(label,pred.ld))
  meas2    <- round(c(ld.p$Accuracy, ld.p$FPR, ld.p$TPR, ld.p$FNR, ld.p$TNR),2)
  meas2a   <- round(c(ld.p$F.measure, ld.p$Precision, ld.p$Recall, 
              ld.p$Specificity, ld.p$Sensitivity),2)

  pred.qd  <- predict(qda(XY[,-(p+1)], XY[,(p+1)]),XY[,-(p+1)])$class
  qd.p     <- measures(label, pred.qd)
  print(table(label,pred.qd))
  meas3    <- round(c(qd.p$Accuracy, qd.p$FPR, qd.p$TPR, qd.p$FNR, qd.p$TNR),2)
  meas3a   <- round(c(qd.p$F.measure, qd.p$Precision, qd.p$Recall, 
              qd.p$Specificity, qd.p$Sensitivity),2)

  meas<-data.frame(rbind(meas1, meas2, meas3))
  colnames(meas)<-c('Acc', 'FPR', 'TPR', 'FNR', 'TNR')
  rownames(meas)<-c('Logistic','Linear','Quadratic')
  
  measa<-data.frame(rbind(meas1a, meas2a, meas3a))
  colnames(measa)<-c('F-measure', 'Precision', 'Recall', 'Specif', 'Sensit')
  rownames(measa)<-c('Logistic','Linear','Quadratic')

  print(meas)

  cat(rep('\n',5))
  print(measa)

  m <- 30     # Number of replications

#
#         Splits formed by
#      Simple Random Sampling 
#

  n  <- nrow(XY)
  p  <- ncol(XY) - 1
  
  prop.split   <- 3/5  # 60% for training 40% for testing
  ptr.0.simple <- numeric(m)

  log.c.tr <- log.c.te <- matrix(0, ncol=10, nrow=m)
  lda.c.tr <- lda.c.te <- matrix(0, ncol=10, nrow=m)
  qda.c.tr <- qda.c.te <- matrix(0, ncol=10, nrow=m)

# Loop from here for replications and generalization

  for(r in 1:m)
  {

    well.defined <- FALSE
    while(!well.defined)
    {
      s     <- sample(sample(sample(n)))
      ntr   <- round(prop.split*n)
      id.tr <- s[1:ntr]
      id.te <- setdiff(s,id.tr)
      nte   <- n-ntr
      id0.tr <- which(XY[id.tr,p+1]==0)
      id1.tr <- which(XY[id.tr,p+1]==1) 
      well.defined <- (det(cov(XY[id1.tr,-(p+1)]))!=0) & (det(cov(XY[id0.tr,-(p+1)]))!=0) 
    }    

    ptr.0.simple[r] <- length(which(XY[id.tr,]$Y == 0))/ntr
     
    label.tr <- XY[id.tr,]$Y
    label.te <- XY[id.te,]$Y

    log.XY <- glm(Y~., family= binomial, data=XY[id.tr,])
    log.XY.f <- step(log.XY, Y~., direction="both", trace=0, k=log(ntr))
    Predicted <- rep(0, ntr)
    Predicted[which(predict(log.XY.f, XY[id.tr,-(p+1)], type = "response") > cutoff)]<-1
    response.tr <- factor(Predicted)
    log.c.tr[r, ] <- extract(measures(label.tr, response.tr))
    Predicted <- rep(0, nte)
    Predicted[which(predict(log.XY.f, XY[id.te,-(p+1)], type = "response") > cutoff)]<-1
    response.te <- factor(Predicted)
    log.c.te[r, ] <- extract(measures(label.te, response.te))
    degenerate <- (length(which(response.tr==1))==0) | (length(which(response.tr==0))==0) 
    rm(response.tr, response.te)
    
    lda.XY  <- lda(XY[id.tr,-(p+1)], XY[id.tr,p+1])
    response.tr  <- predict(lda.XY,XY[id.tr,-(p+1)])$class
    lda.c.tr[r, ] <- extract(measures(label.tr, response.tr))
    response.te <- predict(lda.XY,XY[id.te,-(p+1)])$class
    lda.c.te[r, ] <- extract(measures(label.te, response.te))
    rm(response.tr, response.te)

    qda.XY  <- qda(XY[id.tr,-(p+1)], XY[id.tr,p+1])
    response.tr  <- predict(qda.XY,XY[id.tr,-(p+1)])$class
    qda.c.tr[r, ] <- extract(measures(label.tr, response.tr))
    response.te  <- predict(qda.XY,XY[id.te,-(p+1)])$class
    qda.c.te[r, ] <- extract(measures(label.te, response.te))
    rm(response.tr, response.te)

  }
   
  cat(rep('\n', 126))
  m.tr<-data.frame(rbind(colMeans(log.c.tr), colMeans(lda.c.tr),colMeans(qda.c.tr)))
  m.tr <- round(m.tr,2) 
  colnames(m.tr)<-c('Acc', 'Pre', 'Rec','F.m', 'FPR', 'TPR', 'FNR', 'TNR', 'Spec', 'Sens')
  rownames(m.tr)<-c('Logistic','Linear','Quadratic')
  cat('\nTraining Performances on the ', txtdata, '\n')
  print(m.tr)

  m.te<-data.frame(rbind(colMeans(log.c.te),colMeans(lda.c.te), colMeans(qda.c.te)))
  m.te <- round(m.te,2) 
  colnames(m.te)<-c('Acc', 'Pre', 'Rec','F.m', 'FPR', 'TPR', 'FNR', 'TNR', 'Spec', 'Sens')
  rownames(m.te)<-c('Logistic','Linear','Quadratic')
  cat('\nGeneralization Performances on the ', txtdata, '\n')
  print(m.te) 

# Plot accuracy across methods
  
  accuracy.simple <- matrix(0, nrow=m, ncol=3)

  accuracy.simple[,1]<- log.c.te[,1]
  accuracy.simple[,2]<- lda.c.te[,1]
  accuracy.simple[,3]<- qda.c.te[,1]

  x11()
  boxplot(accuracy.simple, 
  names= c('Logistic', 'LDA', 'QDA'), 
  col=2+(1:5), main = paste('Accuracy comparison for ',txtdata))

# Perform analysis of variance on accuracy
   
  accu.simple <- data.frame(matrix(accuracy.simple, ncol=1),  rep(1:3, each=m))
  colnames(accu.simple)<-c('accuracy','method')
  accu.simple <- data.frame(accu.simple)
 
  cat('\nAnalysis of Variance on Accuracy\n')
  print(summary(aov(accuracy~method, data=accu.simple)))
  
  x11()
  par(mfrow=c(2,2))
  plot(aov(accuracy~method, data=accu.simple))

  m.cost <- matrix(0, ncol=2, nrow=5)

  cost <- function(fpr,fnr)
  {
     return(round(1000*fpr + 5000*fnr))
  } 
 
  for(j in 1:5)
  {
    m.cost[j, 1] <- cost(m.tr[j,5], m.tr[j,7])
    m.cost[j, 2] <- cost(m.te[j,5], m.te[j,7])
  }
  colnames(m.cost)<-c('Training','Test')
  rownames(m.cost)= c('Logistic', 'LDA', 'QDA') 
  print(m.cost)


#
#        Splits formed by
#    Stratified Random Sampling 
#

  id0 <- which(XY$Y == 0)
  id1 <- which(XY$Y == 1)
  n0  <- length(id0)
  n1  <- length(id1)
  n   <- nrow(XY)
  p   <- ncol(XY) - 1

 
  prop0 <- n0/n     # prop0 of any sample must be from 0
  prop1 <- 1-prop0            

  prop.split       <- 3/5 # 60% for training 40% for testing
  ptr.0.stratified <- numeric(m)

  log.c.tr <- log.c.te <- matrix(0, ncol=10, nrow=m)
  lda.c.tr <- lda.c.te <- matrix(0, ncol=10, nrow=m)
  qda.c.tr <- qda.c.te <- matrix(0, ncol=10, nrow=m)


# Loop from here for replications and generalization

  for(r in 1:m)
  {
    well.defined <- FALSE
    while(!well.defined)
    {
      s0     <- sample(sample(id0))
      s1     <- sample(sample(id1)) 
      id0.tr <- s0[1:round(n0*prop.split)]    
      id1.tr <- s1[1:round(n1*prop.split)]
      well.defined <- (det(cov(XY[id1.tr,-(p+1)]))!=0) & (det(cov(XY[id0.tr,-(p+1)]))!=0) 
    }
 
    id0.te <- setdiff(s0, id0.tr) 
    id1.te <- setdiff(s1, id1.tr)

    id.tr  <- union(id0.tr, id1.tr)
    id.te  <- union(id0.te, id1.te)
    ntr    <- length(id.tr)
    nte    <- length(id.te)
    
    ptr.0.stratified[r] <- length(which(XY[id.tr,]$Y == 0))/ntr

    label.tr <- XY[id.tr,]$Y
    label.te <- XY[id.te,]$Y

    log.XY <- glm(Y~., family= binomial, data=XY[id.tr,])
    log.XY.f <- step(log.XY, Y~., direction="both", trace=0,k=log(ntr))
    Predicted <- rep(0, ntr)
    Predicted[which(predict(log.XY.f, XY[id.tr,-(p+1)], type = "response") > cutoff)]<-1
    response.tr <- factor(Predicted)
    log.c.tr[r, ] <- extract(measures(label.tr, response.tr))
    Predicted <- rep(0, nte)
    Predicted[which(predict(log.XY.f, XY[id.te,-(p+1)], type = "response") > cutoff)]<-1
    response.te <- factor(Predicted)
    log.c.te[r, ] <- extract(measures(label.te, response.te))
    rm(response.tr, response.te)

    lda.XY  <- lda(XY[id.tr,-(p+1)], XY[id.tr,p+1])
    response.tr    <- predict(lda.XY,XY[id.tr,-(p+1)])$class
    lda.c.tr[r, ] <- extract(measures(label.tr, response.tr))
    response.te <- predict(lda.XY,XY[id.te,-(p+1)])$class
    lda.c.te[r, ] <- extract(measures(label.te, response.te))
    rm(response.tr, response.te)

    qda.XY  <- qda(XY[id.tr,-(p+1)], XY[id.tr,p+1])
    response.tr  <- predict(qda.XY,XY[id.tr,-(p+1)])$class
    qda.c.tr[r, ] <- extract(measures(label.tr, response.tr))
    response.te  <- predict(qda.XY,XY[id.te,-(p+1)])$class
    qda.c.te[r, ] <- extract(measures(label.te, response.te))
    rm(response.tr, response.te)

  }
   
  cat(rep('\n', 126))
  m.tr<-data.frame(rbind(colMeans(log.c.tr),colMeans(lda.c.tr),colMeans(qda.c.tr)))
  m.tr <- round(m.tr,2) 
  colnames(m.tr)<-c('Acc', 'Pre', 'Rec','F.m', 'FPR', 'TPR', 'FNR', 'TNR', 'Spec', 'Sens')
  rownames(m.tr)<-c('Logistic','Linear','Quadratic')
  cat('\nTraining Performances on the ', txtdata, '\n')
  print(m.tr)

  m.te<-data.frame(rbind(colMeans(log.c.te), colMeans(lda.c.te),colMeans(qda.c.te)))

  m.te <- round(m.te,2) 
  colnames(m.te)<-c('Acc', 'Pre', 'Rec','F.m','FPR', 'TPR', 'FNR', 'TNR','Spec', 'Sens')
  rownames(m.te)<-c('Logistic','Linear','Quadratic')
  cat('\nGeneralization Performances on the ', txtdata, '\n')
  print(m.te) 

# Plot accuracy across methods
  
  accuracy.stratified <- matrix(0, nrow=m, ncol=3)

  accuracy.stratified[,1]<- log.c.te[,1]
  accuracy.stratified[,2]<- lda.c.te[,1]
  accuracy.stratified[,3]<- qda.c.te[,1]

  x11()
  boxplot(accuracy.stratified, 
  names= c('Logistic', 'LDA', 'QDA'), 
  col=2+(1:5), main = paste('Accuracy comparison for ',txtdata))

# Perform analysis of variance on accuracy
   
  accu.stratified <- data.frame(matrix(accuracy.stratified, ncol=1),  rep(1:3, each=m))
  colnames(accu.stratified)<-c('accuracy','method')
  accu.stratified <- data.frame(accu.stratified)
 
  cat('\nAnalysis of Variance on Accuracy\n')
  print(summary(aov(accuracy~method, data=accu.stratified)))
  
  x11()
  par(mfrow=c(2,2))
  plot(aov(accuracy~method, data=accu.stratified))

  methods= c('Logistic', 'LDA', 'QDA')

  x11()
  matplot(cbind(1:3,1:3), 
          cbind(colMeans(accuracy.simple),colMeans(accuracy.stratified)),
          type='l', lty=2:3, col=2:3, xlab='classification method',
          ylab='Accuracy', main='Simple Random Splits vs Stratified Random Splits')  
  legend("bottomleft", c('Simple Random Splits','Stratified Random Splits'),
         lty=2:3, col=2:3)
 
  x11()
  matplot(cbind(1:m,1:m,1:m),
          cbind(ptr.0.simple,ptr.0.stratified, rep(prop0,m)),
          type='l', lwd=c(1,2,2), lty=2:4, col=2:4, xlab='Replicate',
          ylab='Proportion of negative')  
  legend("bottomright", c('Simple','Stratified', 'True'),
         lty=2:4, col=2:4)

#
#  The above two blocks of instructions can be made into functions
#                   for clarity and simplicity
#

#
# Othe activities of great importance 
#

# Error bars

  x11()
  xb <- 1:5; yb<-colMeans(accuracy)
  eb <- 2*apply(accuracy, 2, sd)/50
  plot(xb, yb, type='p', xlab='Classifier')  
  arrows(xb,yb+eb, xb, yb-eb, angle=90, code=3, length=0.05)
  lines(xb, yb, type='l')

# Cross Validation for Tuning

  sc<- sample(sample(n))
  nc <- n
  XYc <- XY[sc[1:nc],]
  cross <- FALSE
  if(cross)
  {
  vC <- seq(1, 15, by=1)
  v.nu <- seq(0.01, 1, lengtn=nC)
  nC <- length(vC)
  v.nu <- seq(0.01, 1, length=nC)
  error <-matrix(0, nrow=nC, ncol=2)

  k   <- round(nc/10)
  S   <- sample(sample(nc))
  c   <- ceiling(nc/k)
  held.out.set <- matrix(0, nrow=c, ncol=k) 
  for(ic in 1:(c-1)){held.out.set[ic,] <- S[((ic-1)*k + 1):(ic*k)]}
  held.out.set[c, 1:(nc-(c-1)*k)] <- S[((c-1)*k + 1):nc]
 
  for(j in 1:nC)
  { 
    for(i in 1:c)
    {   
     out <-  held.out.set[i,] 
     yhatc<-predict(ksvm(Y~., data=XYc[-out,], 
            type='C-svc', C=vC[j],  
            kernel='laplacedot', cross = 10),
            data.frame(XYc[out,-(p+1)]))
    
     yhatn<-predict(ksvm(Y~., data=XYc[-out,], 
            type='nu-svc', nu=v.nu[j],  
            kernel='laplacedot', cross = 10),
            data.frame(XYc[out,-(p+1)]))

     y<- XYc[out,(p+1)]
 
     error[j,1]<-error[j,1] + length(which(y!=yhatc))
     error[j,2]<-error[j,2] + length(which(y!=yhatn))
    }
  }

  error <- error/nc

  x11()
  plot(vC, error[,1], type='l', xlab='C', ylab='CV(C)', 
       main='Cross Validation choice of best C')  
  x11()
  plot(v.nu, error[,2], type='l', xlab=bquote(nu), ylab=paste('CV(',expression(nu),')'), 
       main=paste('Cross Validation choice of best ',expression(nu)))   
  }