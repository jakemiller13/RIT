#
#     Three Simple Functions to Scale the Input Scale
#
#          source('input-scale-1.R')
#
#        Ernest Fokoue (c) Summer 2010
#



#  Standardize: results in mean(x_j)=0  and var(x_j)=1

standardize <-function(xx)
{
  n <- nrow(xx)
  p <- ncol(xx)
  aa  <- matrix(rep(apply(xx,2,mean), n), ncol=p, byrow=TRUE)
  bb  <- matrix(rep(apply(xx,2,sd), n), ncol=p, byrow=TRUE)
  return((xx-aa)/bb)
}

#  Cubitize: results in x_{ij} \in [0,1]

cubitize <-function(xx)
{
  n <- nrow(xx)
  p <- ncol(xx)
  aa  <- matrix(rep(apply(xx,2,min), n), ncol=p, byrow=TRUE)
  bb  <- matrix(rep(apply(xx,2,max), n), ncol=p, byrow=TRUE)
  return((xx-aa)/(bb-aa))
}

#  Unitize: results in mean(x_j)=0 and length(x_j)=1

unitize <-function(xx)
{
  n <- nrow(xx)
  p <- ncol(xx)
  aa  <- matrix(rep(apply(xx,2,mean), n), ncol=p, byrow=TRUE)
  bb  <- sqrt((n-1)*matrix(rep(apply(xx,2,sd), n), ncol=p, byrow=TRUE))
  return((xx-aa)/bb)
}
