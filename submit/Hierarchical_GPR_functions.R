##' Metropolis-Within-Gibbs (MWG) function for the hypermeter parameters
##' 
##' @param n Number of MCMC iterations excluding brun-in 
##' @param burnin Number of brun-in iterstions, Default assumption copied from sample code `i.e. min(nsamples/10, 1000)` 
##' @param y Vector of response variable
##' @param xx Design matrix
##' @param theta Vector of hyperparameters
##' @param mcmc_std Vector of standard deviantion of the proposal distribution for the hyperparameters
##' 
##'    

hyperparameter_MWG <- function(n, burnin, y, xx, theta, mcmc_std){
  
  if(missing(burnin)) burnin <- min(floor(n / 10), 1e3) # default burn-in iterations
  
  # defining output matrix: MCMC results for each iterations
  theta_mwg <- matrix(NA, n, length(theta))  
  colnames(theta_mwg) <- c("beta1", "beta2", "log_Omega11", "log_Omega12", "log_Omega22",
                           "log_nu", "log_tau", "log_kernel width", "log_kernel length")
  
  # definging output variable: acceptance rate
  accr <- rep(0, length(theta))
  names(accr) <- colnames(theta_mwg)
  
  # current position
  theta_curr <- theta
  
  # log-likelihood at current position
  ll_curr <- log_marginal_post(y = y, xx = xx, theta = theta_curr)
  
  for(ii in (-burnin + 1) : n){
    for(jj in 1 : length(theta)){
      
      theta_prop <- theta_curr
      # using random walk for the proposed parameters and updating one parameter at a time
      theta_prop[jj] <- theta_prop[jj] + mcmc_std * rnorm(1)
      
      # log-likelihood with proposed parameters
      ll_prop <- log_marginal_post(y = y, xx = xx, theta = theta_prop)
      
      # log of acceptance probability
      log_accp <- ll_prop - ll_curr
      
      if(log_accp > 0 || runif(1) < exp(log_accp)){
        # conditioned met, and updated the current information with the proposed ones 
        theta_curr <- theta_prop
        ll_curr <- ll_prop
        accr[jj] <- accr[jj] + 1
      }
    }
    # store the results only after burnin
    if(ii > 0) theta_mwg[ii, ] <- theta_curr
  }
  
  # calculating the acceptance rate, and constructing output list
  accr <- accr / (n + burnin)
  out <- list(Theta = theta_mwg, acceptance_rate = accr)
  return(out)
}     


##' Auxiliary function for MCMC
##' This is the function of Marginal Posterior for hyperparameters
##' 
##' @param y Vector of response variable
##' @param xx Design matrix
##' @param theta Vector of hyperparameters
##' @theta[1:2] lambda1-lambda2
##' @theta[3:5] Omega11, Omega12, Omega22
##' @theta[6] nu
##' @theta[7] tau
##' @theta[8] radial kernel width 
##' @theta[9] radial kernel length
##' 

log_marginal_post <- function(y, xx, theta){
  # assume flat prior on hyperparameters
  lambda <- theta[1:2]
  
  # Omega is the precision matrix
  logc_Omega <- matrix(0, 2,2)
  logc_Omega[upper.tri(logc_Omega, diag = TRUE)] <- theta[3 : 5]
  diag(logc_Omega) <- exp(diag(logc_Omega))
  Omega <- crossprod(logc_Omega)
  
  nu <- exp(theta[6])
  tau <- exp(theta[7])
  width <- exp(theta[8])
  length <- exp(theta[9])
  cov_mat <- diag(rep(1, nrow(xx))) + kernel(xx, w = width, l = length)
  
  # hyperparameters for the marginal posterior distibutions 
  Omega_star <- t(xx) %*% solve(cov_mat) %*% xx + Omega
  lambda_star <- solve(Omega_star) %*% (Omega %*% lambda + t(xx) %*% solve(cov_mat) %*% y)
  nu_star <- nrow(xx) + nu
  tau_star <- (t(y) %*% solve(cov_mat) %*% y - t(lambda_star) %*% Omega_star %*% lambda_star +
                 t(lambda) %*% Omega %*% lambda + nu * tau) / nu_star
  
  zeta(Omega_star, nu_star, tau_star) - 
    zeta(Omega, nu, tau) - 0.5 * log(det(cov_mat))
   
}


##' Auxiliary function for MCMC
##' Zeta function - marginal log-likelihood 
##' 
##' @param lambda Vecotr of parameters for MVN
##' @param Omega Covariance matrix for MVN
##' @param nu Parameter for inverse gamma
##' @param tau Parameter for inverse gamma
##' 
zeta <- function(Omega, nu, tau){
  log(gamma(0.5 * nu)) - 0.5 * log(det(Omega)) - 0.5 * nu * log(nu * tau * 0.5)
}

##' Radial Kernel
##'
##' @param w Width parameter
##' @param l length parameter
##'
kernel <- function(data, w = 1, l = 1) {
  #radial kernel, data must be in matrix form
  dist <- as.matrix(dist(data))
  w * exp(-0.5 * dist^2 / l) 
}



##' quantile of Multivariate Normal Inverse Gamma distribution
##'
##' @param n the number of simulations
##' @param lambda
##' @param Omega precision matrix 
##' @param a = 0.5 * nu for inverse scaled chi-square
##' @param b = 0.5 * nu * tau for inverse scaled chi-square
##'

rmNIG <- function(n, lambda, Omega, a, b){
  p <- length(lambda)
  out <- matrix(rep(0, n*(p + 1)), nrow = n)

  for (ii in 1 : n){
    sigma <- 1.0/sqrt(rgamma(n = 1, shape = a, rate = b))
    beta = mvrnorm(1, lambda, sigma^2 * solve(Omega))
    # the following method is identical to mvrnorm function
    # z = rnorm(p, 0, sigma)
    # beta = solve(chol(Omega), z) + lambda
    out[ii, ] <- c(beta, sigma)
  }  
  return(out)
}

##' this function is for the parameters in the joint posterior distribution 
##'
##' @param input the hyperparameters
##' 

hat_estimators <- function(input){
  Omega_star <- t(input$x) %*% solve(input$cov_mat) %*% input$x + input$Omega
  lambda_star <- solve(Omega_star) %*% (input$Omega %*% input$lambda + t(input$x) %*% solve(input$cov_mat) %*% input$y_true)
  nu_star <- nrow(input$x) + input$nu
  tau_star <- (t(input$y_true) %*% solve(input$cov_mat) %*% input$y_true - t(lambda_star) %*% Omega_star %*% lambda_star +
                 t(input$lambda) %*% input$Omega %*% input$lambda + input$nu * input$tau) / nu_star
  
  list(Omega_star = Omega_star, lambda_star = lambda_star, nu_star = nu_star, tau_star = tau_star)
}
