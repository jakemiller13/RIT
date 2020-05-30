#       Stratified Holdout for Comparison of Learners
#
#               source('stratified-holdout-1.R')
#
#                (c) Ernest Fokoue, Summer 2010
#

graphics.off()

stratified.holdout <- function(y, ptr)
{
  n              <- length(y)
  labels         <- unique(y)       # Obtain classifiers
  id.tr <- id.te <- NULL
  # Loop once for each unique label value
  for(j in 1:length(labels)) 
  {
    sj    <- which(y==labels[j])  # Grab all rows of label type j  
    nj    <- length(sj)           # Count of label j rows to calc proportion below
    
    id.tr <- c(id.tr, (sample(sample(sample(sj))))[1:round(nj*ptr)])
  }                               # Concatenates each label type together 1 by 1
  
  id.te  <- (1:n) [-id.tr]          # Obtain and Shuffle test indices to randomize                                
  
  return(list(idx1=id.tr,idx2=id.te)) 
}  


#
# Example
#


library(kernlab)
library(MASS)

XY <- Pima.tr
pos <- ncol(XY)    # Position of the response variable

colnames(XY)[pos] <- 'Y'

R        <- 50
test.err <- matrix(1, nrow=R, ncol=2)
ptr <- 2/3

for(r in 1:R)
{
  
  hold <- stratified.holdout(as.factor(XY[,pos]), ptr) 
  id.tr <- hold$idx1
  id.te <- hold$idx2
  
  ksvm.xy <- ksvm(Y~., data=XY[id.tr,], kernel='rbfdot', type='nu-svc')
  test.err[r,1] <- 1-sum(diag(prop.table(table(XY[id.te,pos],predict(ksvm.xy, XY[id.te,-pos])))))
  
  gauss.xy <- gausspr(Y~., data=XY[id.tr,], kernel='rbfdot')
  test.err[r,2] <- 1-sum(diag(prop.table(table(XY[id.te,pos],predict(gauss.xy, XY[id.te,-pos])))))
  
} 

x11()
boxplot(test.err, names=c('Support Vector Machine', 'Gaussian Process'), main='Test Error comparisons')


  