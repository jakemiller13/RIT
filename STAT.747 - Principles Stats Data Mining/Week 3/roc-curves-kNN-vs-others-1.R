#
#     source('roc-curves-kNN-vs-others-1.R')
#


  graphics.off()

  library(ROCR)
  library(MASS)
  library(e1071)
  library(kernlab)
  library(class)


  name <- '2D Data'


  #  Cubitize
  
  cubitize <-function(xx)
  {
    n <- nrow(xx)
    p <- ncol(xx)
    aa  <- matrix(rep(apply(xx,2,min), n), ncol=p, byrow=TRUE)
    bb  <- matrix(rep(apply(xx,2,max), n), ncol=p, byrow=TRUE)
    return((xx-aa)/(bb-aa))
  }
  
  
  
  xy <- read.csv('german-credit.csv')
  xy <- read.csv('crab.csv')
  
  library(mlbench)
  data(PimaIndiansDiabetes)
  xy <- PimaIndiansDiabetes
  xy[,ncol(xy)] <- ifelse(xy$diabetes=='pos',1,0)
  
  xy <- read.csv('banana-data-1.csv')
  xy <- read.csv('four-corners-data-1a.csv')
  xy <- read.csv('doughnuts.csv')
  xy <- read.csv('doughnuts-easy.csv')

  library(mlbench)
  data(PimaIndiansDiabetes)
  xy <- PimaIndiansDiabetes
  xy[,ncol(xy)] <- ifelse(xy$diabetes=='pos',1,0)
  
  #xy <- read.csv('colon-cancer.csv')
  #xy[,1] <- ifelse(xy[,1]==1,1,0)
  
  #xy <- read.csv('mnist-17.csv', header=FALSE)
  #xy[,1] <- ifelse(xy[,1]==1,1,0)
  
  n  <- nrow(xy)
  p  <- ncol(xy) - 1
  py <- 1+p

# Cubitize the input space whenever possible
  
  xy[,-py] <- cubitize(xy[,-py])
  
  colnames(xy)[py] <- 'y'
  
  lda.xy <- lda(xy[,-py], xy[,py])
  pred.lda <- prediction(predict(lda.xy, xy[,-py])$posterior[,2], xy[,py])
  perf.lda <- performance(pred.lda, measure='tpr', x.measure='fpr')

  qda.xy <- qda(xy[,-py], xy[,py])
  pred.qda <- prediction(predict(qda.xy, xy[,-py])$posterior[,2], xy[,py])
  perf.qda <- performance(pred.qda, measure='tpr', x.measure='fpr')

  kNN.mod <- class::knn(xy[,-py], xy[,-py], xy[,py], k=3, prob=TRUE)
  prob <- attr(kNN.mod, 'prob')
  prob <- 2*ifelse(kNN.mod == "0", 1-prob, prob) - 1

  pred.knn <- prediction(prob, xy[,py])
  perf.knn <- performance(pred.knn, measure='tpr', x.measure='fpr')

  x11()
 
  plot(perf.knn, col=2, lwd= 2, lty=2, main=paste('Comparative ROC curves'))
  plot(perf.qda, col=3, lwd= 2, lty=3, add=TRUE)
  plot(perf.lda, col=4, lwd= 2, lty=4, add=TRUE)
  abline(a=0,b=1)
  legend('bottomright', inset=0.05, c('kNN','QDA', 'LDA'),  col=2:4, lty=2:4)

