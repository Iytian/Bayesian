#include <TMB.hpp> // _must_ put this first
// include losmix library
#include "utils.hpp"
#include "mNIX_GPR.hpp"

template<class Type>
Type objective_function<Type>::operator() () {
	using namespace losmix;
	// --- INPUT VARIABLES ---
	// data input: output of the call losmix::format_data
	DATA_MATRIX(y); // response vector y (as a matrix)
	DATA_MATRIX(Xtr); // _transpose_ of covariate matrix X
	DATA_VECTOR(logC_cov)     // cholesky factor of covariance matrix              //Iytian added covariance matrix

	// parameter input (order is arbitrary)
		PARAMETER_VECTOR(lambda);
	PARAMETER_VECTOR(logC_Omega); // cholesky factor of Omega
	PARAMETER(log_nu);
	PARAMETER(log_tau);

	// --- INTERMEDIATE VARIABLES --- 
	int p = lambda.size(); // number of regression coefficients
	int n = Xtr.cols();

	// hyperparameter conversions of Omega, nu, tau
	matrix<Type> Omega(p, p);
	matrix<Type> cov(n, n);

	// lchol2var expects one-column matrix as second input
	utils<Type>::lchol2var(Omega, logC_Omega.matrix());
	utils<Type>::lchol2var(cov, logC_cov.matrix());

	Type nu = exp(log_nu);
	Type tau = exp(log_tau);
	mNIX<Type> mnix(p, n); // mNIX distribution object								//Iytian added extra input variable to mnix 	


  // --- OUTPUT VARIABLES ---
	Type mll = 0; // marginal log-likelihood
	// random effects variables
	matrix<Type> beta(p, 1); // regression coefficients
	vector<Type> sigma(1); // error standard deviations

  // --- CALCULATIONS ---
	//Iytian --- Removed the for loop from original code
		// posterior calculations
	mnix.set_prior(lambda, Omega, nu, tau);
	mnix.set_suff(y, Xtr, cov);
	mnix.calc_post();
	mll += mnix.log_marg(); // increment marginal likelihood
	SIMULATE{
		// simulate random effect for each subject
		// enclose in SIMULATE block to avoid AD derivative calculations,
		// i.e., faster
		mnix.simulate(beta.col(1), sigma(1));
	}
		// hyperparameter prior
		// the default prior on unconstrained scalars is "uniform(-Inf,Inf)"
		// the default prior on the log-Cholesky decomposition of a variance matrix
		// is the 'lchol_prior' below.
	mll += utils<Type>::lchol_prior(logC_Omega);
	// return random effects
	SIMULATE{
	  REPORT(beta);
	  REPORT(sigma);
	}
	return -mll; // TMB expects a _negative_ log marginal posterior
}
