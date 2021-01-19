import numpy
import numpy.matlib
import matplotlib.pyplot as plt

import scipy.linalg

import sidpy

from sklearn.linear_model import LinearRegression

def var_to_autocov(A, Sigma, nlag=None, return_big_gamma=False):
	"""
	Compute the autocovariance matrices for the VAR process with
	model matrix A and noise variance-covariance matrix Sigma up
	to nlag.

	NOTE: This function is based on the function of the same name
	from the MVGC toolbox for MATLAB:

	https://users.sussex.ac.uk/~lionelb/MVGC/html/mvgchelp.html

	Parameters
	----------
	A : numpy.array
			The (K x K x p) model matrix given the coefficients
			for the VAR(p) process.
	Sigma : numpy.array
			The K x K noise variance-covariance matrix.
	nlag : int
			The number of lags to compute the autocovariance
	return_big_gamma : bool
			Whether or not (default) to return the 
			autocovariance matrix for the stacked VAR(p)
			process used to "start up" the Yule-Walker equations
			when p > 1.


	Returns
	-------
	Gammas : numpy.array
			The K x K x (nlag + 1) matrix giving the
			autocovariance matrices for the VAR(p) model
			up to nlag.
	Gamma0 : numpy.array
			The K*p x K*p autocovariance matrix determined by
			the reverse solution of the Yule-Walker equations
			for the stacked VAR process, whose blocks give
			Gammas[0], Gammas[1], ..., Gammas[p-1].

	Notes
	-----
	See section A.5 of 

		Barnett, L., & Seth, A. K. (2014). The MVGC multivariate Granger causality toolbox: A new approach to Granger-causal inference. Journal of Neuroscience Methods, 223, 50–68. http://doi.org/10.1016/j.jneumeth.2013.10.018
	
	for more details.

	Examples
	--------
	>>> import module_name
	>>> # Demonstrate code here.

	"""

	p = A.shape[2]
	K = A.shape[0]

	# From documentation:
	# Solves the discrete Lyapunov equation \((A'XA-X=-Q)\).
	#		** Actually solves A X A' - X = -Q **

	pK1 = (p-1)*K;

	BOTTOM = numpy.column_stack((numpy.identity(pK1), numpy.zeros((pK1, K))))

	# The stacked model matrix.

	A1 = numpy.row_stack((A.reshape((K, K*p), order = 'F'), BOTTOM))

	acdectol = 1e-8 # The tolerance below which the autocovariance will be considered effectively 0

	rho = numpy.max(numpy.abs(numpy.linalg.eig(A1)[0]))

	acminlags = numpy.ceil(numpy.log(acdectol)/numpy.log(rho))

	if nlag is None:
		nlag = int(acminlags)
	else:
		# nlag = numpy.min([int(acminlags), nlag]) # From original mvgc code
		pass

	TOP = numpy.column_stack((Sigma, numpy.zeros((K, pK1))))
	BOTTOM = numpy.column_stack((numpy.zeros((pK1, K)), numpy.zeros((pK1, pK1))))

	# The stacked noise variance-covariance matrix.

	Sigma1 = numpy.row_stack((TOP, BOTTOM))

	# The autocovariance matrix for the stacked VAR(p) process,
	# giving Gammas[k] along the diagonal, for k = 0, 1, ..., p-1.

	Gamma0 = scipy.linalg.solve_discrete_lyapunov(A1, Sigma1)

	Gammas = numpy.zeros((K, K, nlag))

	Gammas[:, :, :p] = Gamma0[:K, :].reshape((K, K, p), order = 'F')

	# Compute remaining autocovariance using forward Yule-Walker.

	for i in range(p, nlag):
		for j in range(p):
			Gammas[:, :, i] += numpy.matmul(A[:, :, j], Gammas[:, :, i-1-j])

	if return_big_gamma:
		return Gammas, Gamma0
	else:
		return Gammas

def autocov_to_var(GammasSUB, plot=False):
	"""
	Compute the model matrix A and noise variance-covariance matrix
	Sigma from the autocovariance matrices of a VAR(p) process.

	NOTE: This function is based on the function of the same name
	from the MVGC toolbox for MATLAB:

	https://users.sussex.ac.uk/~lionelb/MVGC/html/mvgchelp.html

	Parameters
	----------
	GammasSUB : (K' x K' x q)
			The autocovariance matrices of a subset of a
			VAR(p) process.
	plot : bool
			Whether or not (default) to plot the 
			autocovariance matrices, noise variance-covariance
			matrix, and model matrix for the subset of the
			VAR(p) process.

	Returns
	-------
	ASUB : numpy.array
			The K' x K' x q model matrix for the subprocess
			of the VAR(p) model.
	SigmaSUB : numpy.array
			The K' x K' noise variance-covariance matrix
			for the subprocess of the VAR(p) model.

	Notes
	-----
	Any notes go here.

	Examples
	--------
	>>> import module_name
	>>> # Demonstrate code here.

	"""

	q = GammasSUB.shape[2]-1 # Not sure about this

	Ksub = GammasSUB.shape[0]

	GamL = GammasSUB[:, :, 1:(q+1)].reshape((Ksub, Ksub*q), order = 'F')

	GamR = numpy.zeros((Ksub*q, Ksub*q))

	for i in range(0, q-1):
		for j in range(i+1, q):
			GamR[i*Ksub:(i+1)*Ksub, j*Ksub:(j+1)*Ksub] = GammasSUB[:, :, j-i]

	GamR = GamR + GamR.T

	for i in range(0, q):
		GamR[i*Ksub:(i+1)*Ksub, i*Ksub:(i+1)*Ksub] = GammasSUB[:, :, 0]

	if plot:
		plt.figure()
		plt.imshow(GamR)

	ASUB = numpy.linalg.solve(GamR.T, GamL.T).T
	SigmaSUB = GammasSUB[:, :, 0] - numpy.matmul(ASUB, GamL.T)

	if plot:
		plt.figure()
		plt.imshow(SigmaSUB)

	if plot:
		fig, ax = plt.subplots(Ksub, 1, sharex = True)
		if Ksub == 1:
			ax.plot(ASUB.squeeze())
		else:
			for i in range(Ksub):
				ax[i].plot(ASUB[i, :])

	return ASUB.reshape(Ksub, Ksub, q), SigmaSUB

def tsdata_to_infocrit(X, morder, verbose = False):
	"""
	Compute the AIC and BIC for a Gaussian VAR(p)
	model as a function of p up to p = morder using
	the recursive algorithm given in

	Martin Morf, Augusto Vieira, Daniel TL Lee, and Thomas Kailath. 
	"Recursive multichannel maximum entropy spectral estimation." 
	IEEE Transactions on Geoscience Electronics 16, no. 2 (1978): 85-94.

	given a data array X.

	NOTE: This function is based on the function of the same name
	from the MVGC toolbox for MATLAB:

	https://users.sussex.ac.uk/~lionelb/MVGC/html/mvgchelp.html

	Parameters
	----------
	X : numpy.array
			The K x T data array for the K time series.
	morder : int
			The largest model order to use.
	verbose : bool
			Whether or not (default) to print the current
			model order.

	Returns
	-------
	aic : numpy.array
			The Akaike Information Criterion (AIC) for
			the Gaussian VAR(p) model.

			NOTE: aic[0] corresponds to p = 1, etc.

	bic : numpy.array
			Schwarz's Bayesian Information Criterion (BIC)
			for the Gaussian VAR(p) model.

			NOTE: bic[0] corresponds to p = 1, etc.

	Notes
	-----
	Any notes go here.

	Examples
	--------
	>>> import module_name
	>>> # Demonstrate code here.

	"""

	n, m = X.shape

	X = X - X.mean(1)[:, numpy.newaxis] # Remove mean from each time series

	##% store lags

	q = morder
	q1 = q+1
	q1n = q1*n

	XX = numpy.zeros((n, q1, m+q))
	for k in range(q+1):
		XX[:, k, k:k+m] = X

	aic = numpy.zeros(q)*numpy.nan
	bic = numpy.zeros(q)*numpy.nan
	nll = numpy.zeros(q)*numpy.nan

	I = numpy.identity(n)

	#% initialise recursion

	AF = numpy.zeros((n, q1n)) # forward  AR coefficients
	AB = numpy.zeros((n, q1n)) # backward AR coefficients (reversed compared with Morf's treatment)

	k  = 1 # model order is k-1
	kn = k*n
	M  = (m - k)
	kf = numpy.arange(kn)		# forward  indices
	kb = numpy.arange(q1n-kn, q1n) # backward indices

	XF = XX[:, :k, k:m].reshape((kn, M), order = 'F')
	XB = XX[:, :k, (k-1):(m-1)].reshape((kn, M), order = 'F')

	# Note that MATLAB and numpy.linalg return the **transpose** of their
	# respective Cholesky decompositions. (Of course that would happen...)

	CXF = numpy.linalg.cholesky(numpy.matmul(XF, XF.T)).T
	CXB = numpy.linalg.cholesky(numpy.matmul(XB, XB.T)).T

	AF[:, kf] = numpy.linalg.solve(CXF.T, I)
	AB[:, kb] = numpy.linalg.solve(CXB.T, I)

	# See section "SQUARE-ROOT NORMALIZED MEM ALGORITHM" on page 88 of
	#  "Recursive Multichannel Maximum Entropy Spectral Estimation"

	while k <= q:
		if verbose:
			print("On q = {} of {}".format(k, q))

		EF = numpy.matmul(AF[:,kf], numpy.reshape(XX[:,:k,k:m],(kn,M), order = 'F')) #% forward  prediction errors
		EB = numpy.matmul(AB[:,kb], numpy.reshape(XX[:,:k,k-1:m-1],(kn,M), order = 'F')) #% forward  prediction errors

		CEF = numpy.linalg.cholesky(numpy.matmul(EF, EF.T)).T
		CEB = numpy.linalg.cholesky(numpy.matmul(EB, EB.T)).T

		# NOTE: In MATLAB, "A\B" solves for x in Ax = B
		#	   and is equivalent to numpy.linalg.solve(A, B)

		# NOTE: In MATLAB, "B/A" solves for x in xA = B
		#	   and is (presumably) equivalent to...

		# R = CEF'\(EF*EB')/CEB;	   % normalised reflection coefficients

		# Equation 20 from "Recursive Multichannel Maximum Entropy Spectral Estimation"

		# From:
		#   https://research.wmz.ninja/articles/2018/10/notes-on-migrating-doa-tools-from-matlab-to-python.html
		# 
		# "A \ B" = "numpy.linalg.lstsq(A, B)[0]"		  = "numpy.linalg.solve(A, B)"
		# "A / B" = "numpy.linalg.lstsq(B.T, A.T)[0].T" = "numpy.linalg.solve(B.T, A.T).T"

		def mldivide(A, B):
			return numpy.linalg.lstsq(A, B, rcond = None)[0]

		def mrdivide(A, B):
			return numpy.linalg.lstsq(B.T, A.T, rcond = None)[0].T

		R = mrdivide(mldivide(CEF.T, numpy.matmul(EF, EB.T)), CEB)

		CRF = numpy.linalg.cholesky(I - numpy.matmul(R, R.T)).T
		CRB = numpy.linalg.cholesky(I - numpy.matmul(R.T, R)).T

		k  = k+1;
		kn = k*n;
		M  = (m-k);
		kf = numpy.arange(kn)		# forward  indices
		kb = numpy.arange(q1n-kn, q1n) # backward indices

		AFPREV = AF[:, kf]
		ABPREV = AB[:, kb]

		AF[:, kf] = mldivide(CRF.T, AFPREV - numpy.matmul(R,   ABPREV))
		AB[:, kb] = mldivide(CRB.T, ABPREV - numpy.matmul(R.T, AFPREV))

		E = numpy.matmul(mldivide(AF[:, :n], AF[:, kf]), numpy.reshape(XX[:, :k, k:m], (kn, M), order = 'F'))
		
		DSIG = numpy.linalg.det(numpy.matmul(E, E.T)/(M-1))

		i = k-1

		def compute_aic(L, k, m):
			if m-k-1 <= 0:
				return numpy.nan
			else:
				return(-L + k*(m/float(m-k-1)))

		def compute_bic(L, k, m):
			return(-L + 0.5*k*numpy.log(m))

		L = -(M/2.)*numpy.log(DSIG)

		nll[i-1] = -L
		aic[i-1] = compute_aic(L, i*n*n, M)
		bic[i-1] = compute_bic(L, i*n*n, M)

	return aic, bic

def tsdata_to_granger(x, y, p):
	# Currently only for two time series:
	K = 2

	X = sidpy.embed_ts(x, p)
	Y = sidpy.embed_ts(y, p)

	target = numpy.column_stack((X[:, -1], Y[:, -1]))

	XY = numpy.column_stack((X[:, :-1][:, ::-1], Y[:, :-1][:, ::-1]))

	reg = LinearRegression(fit_intercept = False).fit(XY, target)

	target_lm = reg.predict(XY)

	x_var = target_lm[:, 0]
	y_var = target_lm[:, 1]

	resids = numpy.column_stack((X[:, -1] - x_var, Y[:, -1] - y_var))

	A_hat = reg.coef_.reshape((K, K, p))
	Sigma_hat = numpy.cov(resids.T, bias = True)

	Gammas = var_to_autocov(A_hat, Sigma_hat)

	F = [0, 0]

	for i, submodel_inds in enumerate([[0], [1]]):
		Ksub = len(submodel_inds) # Size of a block along one dimension

		GammasSUB = Gammas[numpy.ix_(submodel_inds, submodel_inds)]

		ASUB, SigmaSUB = autocov_to_var(GammasSUB)

		F[i] = numpy.log(numpy.linalg.det(SigmaSUB)) - numpy.log(numpy.linalg.det(Sigma_hat[numpy.ix_(submodel_inds, submodel_inds)]))

	return F, A_hat, Sigma_hat

def tsdata_to_A_and_Sigma(x, p):
	"""
	Estimate the model matrix A and variance-covariance matrix
	Sigma for the VAR(p) model from the time series stored in x
	using Ordinary Least Squares.

	NOTE: The time series are mean-centered, so the intercepts
	are assumed to be 0.

	Parameters
	----------
	x : numpy.array
			The time series stored as a K x T array,
			with K time series each of length T.
	p : int
			The model order for the VAR(p) model.

	Returns
	-------
	A_hat : numpy.array
			The estimate of the K x K x p model matrix.

	Sigma_hat : numpy.array
			The estimate of the K x K noise 
			variance-covariance matrix.

	Notes
	-----
	Any notes go here.

	Examples
	--------
	>>> import module_name
	>>> # Demonstrate code here.

	"""

	# Number of time series.

	K = x.shape[0]

	# Length of time series.

	T = x.shape[1]

	# Convert to 0 mean:

	x = x - x.mean(1).reshape(-1, 1)

	# X ~  K x (T - p) x (p + 1)

	X = sidpy.embed_ts_multvar(x, p)

	# Ultimate dimensions for future and past:

	# future = numpy.zeros((T - p, K))
	# past = numpy.zeros((T - p, K*p))

	future = X[:, :, -1].T

	past = X[:, :, 0:p][:, :, ::-1] # reverse, so A is in 
									# the standard order.

	# Flatten past matrix so it is 
	# 
	# 	(T - p) x (K*p),
	# 
	# as LinearRegression expects, instead of
	# 
	# 	K x (T - p) x p
	# 
	# which is the shape generated by embed_ts_multvar.

	# WRONG, fixed 150121: past = past.reshape((K*p, T - p), order = 'F').T

	past = past.swapaxes(0, 1).swapaxes(1, 2).reshape((T-p, K*p), order = 'F')

	# This assumes time series have been **centered** to have mean 0!
	reg = LinearRegression(fit_intercept = False).fit(past, future)

	future_lm = reg.predict(past)

	resids = future - future_lm

	A_hat = reg.coef_.reshape((K, K, p))
	Sigma_hat = numpy.cov(resids.T, bias = True)

	return A_hat, Sigma_hat

def datamatrix_to_A_and_Sigma(X):
	"""
	Estimate the model matrix A and variance-covariance matrix
	Sigma for the VAR(p) model from the data matrix stored in X
	using Ordinary Least Squares.

	NOTE: The time series are assumed to be mean-centered, so the
	intercepts are assumed to be 0.

	Parameters
	----------
	X : numpy.array
			K x (T - p) x (p + 1) data matrix.
	p : int
			The model order for the VAR(p) model.

	Returns
	-------
	A_hat : numpy.array
			The estimate of the K x K x p model matrix.

	Sigma_hat : numpy.array
			The estimate of the K x K noise 
			variance-covariance matrix.

	Notes
	-----
	Any notes go here.

	Examples
	--------
	>>> import module_name
	>>> # Demonstrate code here.

	"""

	# X ~  K x (T - p) x (p + 1)
	
	# The number of time series.

	K = X.shape[0]

	# The model order.

	p = X.shape[2] - 1

	# The number of time points in the original
	# time series.

	T = X.shape[1] + p

	# Ultimate dimensions for future and past:

	# future = numpy.zeros((T - p, K))
	# past = numpy.zeros((T - p, K*p))

	future = X[:, :, -1].T

	past = X[:, :, 0:p][:, :, ::-1] # reverse, so A is in 
									# the standard order.

	# Flatten past matrix so it is 
	# 
	# 	(T - p) x (K*p),
	# 
	# as LinearRegression expects, instead of
	# 
	# 	K x (T - p) x p
	# 
	# which is the shape generated by embed_ts_multvar.

	# WRONG, fixed 150121: past = past.reshape((K*p, T - p), order = 'F').T

	past = past.swapaxes(0, 1).swapaxes(1, 2).reshape((T-p, K*p), order = 'F')

	# This assumes time series have been **centered** to have mean 0!
	reg = LinearRegression(fit_intercept = False).fit(past, future)

	future_lm = reg.predict(past)

	resids = future - future_lm

	A_hat = reg.coef_.reshape((K, K, p))
	Sigma_hat = numpy.cov(resids.T, bias = True)

	return A_hat, Sigma_hat