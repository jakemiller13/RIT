library(optimbase)

# Problem 1
X <- matrix(c(-8, -4, 0, 4, 8))
X <- cbind(ones(length(X), 1), X)
Y <- matrix(c(11.7, 11, 10.2, 9, 7.8))

################
# Calculations #
################

# Constants
alpha <- 0.05
n <- length(Y)
t_stat <- qt(alpha/2, length(Y) - 2)

# YOU ARE HERE T STATS ARE EQUIVALENT