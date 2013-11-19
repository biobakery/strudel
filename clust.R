# clustered data generation 

## normal/inverse gamma cluster generation 

require(pscl)

num.clusters 	<- 3
num.draws		<- 30

vec.out <- NULL

## First Case 

hyper.alpha <- 1 
hyper.beta  <- 1

prior.theta <- rigamma(1, hyper.alpha, hyper.beta)

vec.out <- c( vec.out, rnorm(num.draws, 0, prior.theta) ) 

## Second Case

hyper.alpha <- 3 
hyper.beta  <- 1

prior.theta <- rigamma(1, hyper.alpha, hyper.beta)

vec.out <- c( vec.out, rnorm(num.draws, 0, prior.theta) ) 

## Third Case

hyper.alpha <- 3 
hyper.beta  <- 0.5

prior.theta <- rigamma(1, hyper.alpha, hyper.beta)

vec.out <- c( vec.out, rnorm(num.draws, 0, prior.theta) ) 

