# loading the functions
source("R_code_functions.R")
compile("ModelExt_GPR_test.cpp")
dyn.load(dynlib("ModelExt_GPR_test"))

##  simulation
# defining the hyperparameters
p <- 2 #  beta1, beta2
nsamples <- 10

set.seed(10)
# hyperparameters for beta
lambda <- rnorm(p)
Omega <- diag(p)  # precision matrix
chol_Omeg <- chol(Omega) # using cholesky factors for MCMC simulations
chol_Omeg <- chol_Omeg[upper.tri(chol_Omeg, diag = TRUE)]

# hyperparameters for sigma
nu <- 5 * runif(1) + 3
tau <- rexp(1) / 100

# generate sigma and beta
sigma <- 1 / sqrt(rgamma(0.5 * nu, 2 / (nu * tau)))
beta <- mvrnorm(1, lambda, sigma^2 * solve(Omega))

# hyperparameters for kernel matrix
k_w <- runif(1, 0.5, 1.5) / sigma
k_l <- runif(1, 1.5, 2.5)

# simulated design matrix
x <- matrix(rnorm(nsamples * p), ncol = p, nrow = nsamples)

f_x <- x %*% beta # mean for response variable - y

# covariance matrix for response variable - y
cov_mat <- diag(rep(1, nsamples)) + kernel(x, w = k_w, l = k_l)

# simulated response variable
y_true <- as.matrix(mvrnorm(1, f_x, sigma^2 * cov_mat))


###   1. test the likelihood function

##  it should give higher value for under the true parameters
sim_theta <- function() {
  lambda0 <- rnorm(2)
  Omega0 <- crossprod(matrix(rnorm(4), 2, 2))
  chol_Omeg0 <- chol(Omega0)
  chol_Omeg0 <- chol_Omeg0[upper.tri(chol_Omeg0, diag = TRUE)]
  
  nu0 <- runif(1, 1, 2)
  tau0 <- rexp(1)/5
  
  k_w0 <- runif(1, 1, 5)
  k_l0 <- runif(1, 1, 5)
  
  c(lambda0, chol_Omeg0, log(nu0), log(tau0), log(k_w0), log(k_l0))
}

theta_t <- c(lambda, chol_Omeg[1], chol_Omeg[2], chol_Omeg[3], log(nu), log(tau), log(k_w), log(k_l))

replicate(50, expr = {
  theta <- sim_theta()
  (log_marginal_post(y_true, x, theta_t) - log_marginal_post(y_true, x, theta) > 0 )
})


###   2. test MCMC function (Metropolis-Within-Gibbs)
##  using MWG method simulate the hyperparameters using the log of marginal posterior distribution

mwg_out <- hyperparameter_MWG(n = 20, y = y_true, xx = x, theta = theta_t, mcmc_std = 1)
# issue: the alogrithm will not converge

# $Theta
# beta1       beta2      Omega11   Omega12   Omega22         nu       tau kernel width kernel length
# [1,]  0.06526557 -0.18425254 -1.284079611  1.191752  3.602169 -1.4112027 -2.662482     1.480152   1.000693263
# [2,]  0.06526557  0.11117343 -0.671524030  2.197934  4.895313 -1.4697396 -3.709366     1.501151   1.000693263
# [3,]  0.06526557  0.11117343  0.745001062  1.647497  5.619537 -1.4697396 -2.604690     3.252793   1.170944492
# [4,]  0.06526557  0.11117343  2.087929370  3.488659  5.617402 -1.4697396 -2.604690     4.032240   0.310273360
# [5,]  0.06526557  0.11117343  1.931732693  3.883382  7.637342 -1.6587124 -3.789064     4.032240   0.005141578
# [6,]  0.06526557  0.23049187  1.494894806  3.749845  7.054391 -0.7768985 -3.789064     4.032240  -1.211206825
# [7,]  0.06526557  0.23049187  0.955517437  3.351066  6.967028 -0.7768985 -3.985781     5.031543  -1.211206825
# [8,]  0.29292956 -0.19673877  1.291449403  4.851754  5.765133 -1.2651748 -3.985781     5.031543  -1.163611504
# [9,]  0.12260338 -0.19673877  0.004612658  4.658999  6.722871 -1.0941769 -4.216930     4.509168  -1.702651856
# [10,] -0.48427550 -0.19673877  0.004612658  5.391495  8.252719 -2.5443271 -3.205771     4.509168  -1.702651856
# [11,]  0.26793357 -0.19673877 -1.438514130  6.718489  7.375518 -2.5443271 -4.189468     4.644079  -1.850197285
# [12,]  0.53407300 -0.19673877 -0.354025710  8.430253  8.472573 -1.6247104 -6.194141     4.644079  -3.844896585
# [13,]  0.53407300 -0.02674383  0.628900036  8.640772  8.889124 -1.6247104 -5.702843     4.780316  -5.305500368
# [14,]  0.53407300 -0.02674383  1.164363033 10.267293  9.326881 -1.6247104 -5.702843     4.617524  -3.493960735
# [15,]  0.53407300 -0.02674383  2.340910408  9.301789 10.609771 -0.8528130 -6.043192     6.835371  -4.558003951
# [16,]  0.42467982 -0.02674383  2.732858595 10.267916 11.667030 -1.2378729 -6.686369     6.835371  -4.698901597
# [17,]  0.42467982 -0.02674383  3.999531601 10.696973  9.759658 -1.3640947 -6.686369     7.135312  -5.990530001
# [18,]  0.42467982 -0.02674383  5.384923582 10.697464  9.383647 -2.4676498 -6.666169     7.839675  -8.042468596
# [19,]  0.42467982 -0.02674383  5.030975415 10.865261  9.640293 -2.4676498 -6.666169    10.144797  -5.309532587
# [20,]  0.42467982  0.28688236  6.453654391 10.755416  9.457645 -2.3992829 -8.219651    10.144797  -5.888760079
# 
# $acceptance_rate
# beta1         beta2       Omega11       Omega12       Omega22            nu           tau  kernel width kernel length 
# 0.3181818     0.2272727     0.9090909     1.0000000     0.9545455     0.6363636     0.7272727     0.6818182     0.8181818 



###   3. test C++ code for simulating the conditional posterior distibution

# log of cholesky decomposition factors for covariance matrix
logc <- chol(cov_mat)
diag(logc) <- log(diag(logc))
logc <- logc[upper.tri(logc, diag = TRUE)]

# log of cholesky decomposition factors for hyperparameter Omega
logC_Omega <- chol(Omega)
diag(logC_Omega) <- log(diag(logC_Omega))
logC_Omega <- logC_Omega[upper.tri(logC_Omega, diag = TRUE)]


cond_post_test <- MakeADFun(data = list(y = y_true, Xtr = t(x), logC_cov = logc), 
                      parameters = list(lambda = lambda, logC_Omega = logC_Omega, log_nu = log(nu),
                                        log_tau = log(tau)), DLL = "ModelExt_GPR_test")
# issue fatal error.... R terminated
cond_post_test$simulate()
