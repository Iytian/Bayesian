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
	DATA_MATRIX(dist2);																//Iytian data input squared distance matrix - radial kernel
	
	// In this forced version moved these two set of hyperparameters to data
	DATA_VECTOR(logC_Omega);	// cholesky factor of Omega												
	DATA_SCALAR(log_nu);


	// parameter input (order is arbitrary)
	PARAMETER_VECTOR(lambda);
	//PARAMETER_VECTOR(logC_Omega);	// cholesky factor of Omega
	//PARAMETER(log_nu);
	PARAMETER(log_tau);
	PARAMETER(log_ww);					// kernel width hyperparameter			
	PARAMETER(log_ll);					// kernel length hyperparapeter

	// --- INTERMEDIATE VARIABLES --- 
	int p = lambda.size(); // number of regression coefficients
	int n = Xtr.cols();
	Type ww = exp(log_ww);
	Type ll = exp(log_ll);
	Eigen::LLT<Eigen::Matrix<Type, Eigen::Dynamic, Eigen::Dynamic> > lltdet_;

	// hyperparameter conversions of Omega, nu, tau
	matrix<Type> Omega(p, p);
	matrix<Type> cov(n, n);
	matrix<Type> cov1(n, n);

	// lchol2var expects one-column matrix as second input
	utils<Type>::lchol2var(Omega, logC_Omega.matrix());

	//Iytian constructing the covariance matrix using the length and width parameters
	cov.setIdentity();
	matrix<Type> tmp = -0.5 * dist2 / ll;
	cov1 = exp(tmp.array()) * ww;
	cov += cov1;

	Type half_ldcov = 0.0;															//Iytian calculating log of determinant of the cov matrix
	for (int ii = 0; ii < n; ii++) {												//Iytian this will be used for the marginal post. dist.
		half_ldcov += log(lltdet_.compute(cov).matrixL()(ii, ii));
	}

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
	mll += mnix.log_marg() - half_ldcov;										//Iytian zeta functions plus half of log det.
	SIMULATE{
		// simulate random effect for each subject
		// enclose in SIMULATE block to avoid AD derivative calculations,
		// i.e., faster
		mnix.simulate(beta.col(0), sigma(0));
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