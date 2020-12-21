import numpy
import numpy.matlib
import matplotlib.pyplot as plt

import scipy.linalg

import ipdb

import sidpy

from sklearn.linear_model import LinearRegression

plt.ion()

def var_to_autocov(A, Sigma, nlag = None):
	p = A.shape[2]
	K = A.shape[0]

	# From documentation:
	# Solves the discrete Lyapunov equation \((A'XA-X=-Q)\).
	#		** Actually solves A X A' - X = -Q **

	pK1 = (p-1)*K;

	BOTTOM = numpy.column_stack((numpy.identity(pK1), numpy.zeros((pK1, K))))

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

	Sigma1 = numpy.row_stack((TOP, BOTTOM))

	Gamma0 = scipy.linalg.solve_discrete_lyapunov(A1, Sigma1)

	Gammas = numpy.zeros((K, K, nlag))

	Gammas[:, :, :p] = Gamma0[:K, :].reshape((K, K, p), order = 'F')

	for i in range(p, nlag):
		for j in range(p):
			Gammas[:, :, i] += numpy.matmul(A[:, :, j], Gammas[:, :, i-1-j])

	return Gammas

def autocov_to_var(GammasSUB):
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

	plt.figure()
	plt.imshow(GamR)

	ASUB = numpy.linalg.solve(GamR.T, GamL.T).T
	SigmaSUB = GammasSUB[:, :, 0] - numpy.matmul(ASUB, GamL.T)

	plt.figure()
	plt.imshow(SigmaSUB)

	fig, ax = plt.subplots(Ksub, 1, sharex = True)
	if Ksub == 1:
		ax.plot(ASUB.squeeze())
	else:
		for i in range(Ksub):
			ax[i].plot(ASUB[i, :])

	return ASUB.reshape(Ksub, Ksub, q), SigmaSUB

def tsdata_to_infocrit(X, morder, verbose = False):
	n, m = X.shape

	# ipdb.set_trace()

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

	## NOTE: This is where the wheels may begin to come off...

	# ipdb.set_trace()

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

		nll[i-1] = (M/2.)*numpy.log(DSIG)
		aic[i-1] = compute_aic(L, i*n*n, M)
		bic[i-1] = compute_bic(L, i*n*n, M)

		# print(aic[i-1]*2, bic[i-1]*2, nll[i-1])
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
	K = x.shape[0]

	T = x.shape[1]

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

	past = past.reshape((K*p, T - p), order = 'F').T

	# This assumes time series have been **centered** to have mean 0!
	reg = LinearRegression(fit_intercept = False).fit(past, future)

	future_lm = reg.predict(past)

	resids = future - future_lm

	A_hat = reg.coef_.reshape((K, K, p))
	Sigma_hat = numpy.cov(resids.T, bias = True)

	return A_hat, Sigma_hat

def datamatrix_to_A_and_Sigma(X):
	# X ~  K x (T - p) x (p + 1)
	
	K = X.shape[0]

	p = X.shape[2] - 1

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

	past = past.reshape((K*p, T - p), order = 'F').T

	# This assumes time series have been **centered** to have mean 0!
	reg = LinearRegression(fit_intercept = False).fit(past, future)

	future_lm = reg.predict(past)

	resids = future - future_lm

	A_hat = reg.coef_.reshape((K, K, p))
	Sigma_hat = numpy.cov(resids.T, bias = True)

	return A_hat, Sigma_hat