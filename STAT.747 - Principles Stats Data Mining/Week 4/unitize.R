#
#  Simple Function for Transforming Variables to  
#           Zero Mean and Unit Length
#
#            source('unitize.R')
#
#            spring 2010, EPF
#


unitize <-function(xx)
{
   n <- nrow(xx)
   p <- ncol(xx)
   aa  <- matrix(rep(apply(xx,2,mean), n), ncol=p, byrow=TRUE)
   bb  <- sqrt((n-1)*matrix(rep(apply(xx,2,var), n), ncol=p, byrow=TRUE))
   return((xx-aa)/bb)
}

# End
