#
#     source('roc-curves-compare-1.R')
#


  graphics.off()

  library(ROCR)
  library(MASS)
  library(e1071)
  library(kernlab)
  library(class)


  name <- '2D Data'


  
  xy <- read.csv('doughnuts-easy.csv')
  xy <- read.csv('doughnuts.csv')

  xy <- Pima.tr
   
  n  <- nrow(xy)
  p  <- ncol(xy) - 1
  py <- p+1
  
  colnames(xy)[py] <- 'y'
  
  lda.xy <- lda(xy[,-py], xy[,py])
  pred.lda <- prediction(predict(lda.xy, xy[,-py])$posterior[,2], xy[,py])
  perf.lda <- performance(pred.lda, measure='tpr', x.measure='fpr')

  qda.xy <- qda(xy[,-py], xy[,py])
  pred.qda <- prediction(predict(qda.xy, xy[,-py])$posterior[,2], xy[,py])
  perf.qda <- performance(pred.qda, measure='tpr', x.measure='fpr')

  pred.knn <- prediction(1-attr(knn(xy[,-py], xy[,-py], xy[,py], k=4, prob=T), 'prob'), xy[,py])
  perf.knn <- performance(pred.knn, measure='tpr', x.measure='fpr')

  quartz()
 
  plot(perf.lda, col=2, lwd= 2, lty=2, main=paste('ROC curves on the Pima Indian Data'))
  plot(perf.qda, col=3, lwd= 2, lty=3, add=TRUE)
  plot(perf.knn, col=4, lwd= 2, lty=4, add=TRUE)
  abline(a=0,b=1)
  legend('bottomright', col=2:4, c('LDA','QDA', 'kNN'), inset=0.05, lty=2:3)

