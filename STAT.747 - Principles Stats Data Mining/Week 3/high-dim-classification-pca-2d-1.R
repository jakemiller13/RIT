#
#  source('high-dim-classification-pca-2d-1.R')
#

   library(ElemStatLearn)
 
 
 
   nte <- nrow(zip.test)
   some <- 540
     
   some.id <- sample(sample(sample(nte)))[1:180]
   z <- summary(princomp(zip.test[,-1]))$scores[,1:2]
   y <- zip.test[some.id,1]
 
   z1 <- z[some.id,1]
   z2 <- z[some.id,2]
 
   digit.2d <- data.frame(z1,z2,y)
   
   ggplot(digit.2d)+
   geom_text(aes( z1 , z2 , label = y , colour = factor(y)))+
   labs(x=expression(z[1]), y=expression(z[2]))+
   theme(legend.position="none") 

# Plot some images of digits and see

   image(matrix(rnorm(256),16,16), col=gray(256:0/256), xaxt= "n", yaxt= "n")

   image(matrix(rnorm(256),16,16), xaxt= "n", yaxt= "n")
   
# It is important for the data from library(ElemStatLearn) to remember zip2image
   
#  Color
   
   par(mfrow=c(3,3))
   some <- sample(1:nte, 9)
   for(i in 1:9)
   { 
     img <- zip2image(zip.test, some[i]) 
     image(img, xaxt= "n", yaxt= "n")
   } 
   
#  Grayscale
   
   par(mfrow=c(3,3))
   some <- sample(1:nte, 9)
   for(i in 1:9)
   { 
     img <- zip2image(zip.test, some[i]) 
     image(img, col=gray(256:0/256), xaxt= "n", yaxt= "n")
   } 
   
   image(matrix(zip.test[165,-1], nrow=16, ncol=16), xaxt="n", yaxt="n", main="1")
   
#  Some faces too
   
   orl <- as.matrix(read.csv('orl-faces2.csv'))
   n.orl <- nrow(orl)
   
   t(matrix(faces[,j], nrow=28, ncol=23))
   

   img <- t(matrix((orl[sample(n.orl)[1],]), nrow=28, ncol=23)) 
   image(img, col=gray(256:0/256), xaxt= "n", yaxt= "n", useRaster=T)
   
   par(mfrow=c(3,3))
   some <- sample(1:n.orl, 9)
   for(i in 1:9)
   { 
     img <- matrix(orl[some[i],], byrow=F, nrow=28, ncol=23) 
     image(t(img), col=gray(256:0/256), xaxt= "n", yaxt= "n")
   }
   
   
   