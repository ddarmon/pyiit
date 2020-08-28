import pymvgc
import numpy

from itertools import combinations

import ipdb

def integrated_synergy(A, Sigma, partition, nlag = 2):
	K = A.shape[0]

	all_nodes = list(range(K))

	#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	#
	# Get variance-covariance matrices:
	#
	#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	Gam = pymvgc.var_to_autocov(A, Sigma, nlag = nlag)

	# Fairly certain this is correct:

	# Var(X_{t})

	SigX0  = Gam[:, :, 0]

	# Cov(X_{t}, X_{t - 1})

	CovX0X1 = Gam[:, :, 1]

	# Fairly certain this is incorrect:

	# # Var(X_{t})

	# SigX0  = Gam[:, :, 0].T

	# # Cov(X_{t}, X_{t - 1})

	# CovX0X1 = Gam[:, :, 1].T

	#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	#
	# Compute I[X_{t-1}; X_t]
	#
	#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	# I[X_{t-1}; X_t] = 0.5 log |SigX0|/|SigX0gX1|

	# SigX0gX1 = SigX0 - CovX0X1*inv(SigX1)*CovX1X0
	# 
	# where SigX1 = SigX0 by wide-sense stationarity.

	# Note: Computing inv(SigX1)*CovX1X0 by solving
	# 
	# 	SigX0*A = CovX10

	SigX0gX1 = SigX0 - CovX0X1 @ numpy.linalg.solve(SigX0, CovX0X1.T)

	IX0X1 = 0.5*numpy.log(numpy.linalg.det(SigX0)/numpy.linalg.det(SigX0gX1))

	#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	#
	# Compute I[M_{t-1}^{(r)}; X_{t}] for each r.
	#
	#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	# I[M_{t-1}^{(r)}; X_{t}] = 0.5 log |SigX0|/|SigX0gM1|

	# SigX0gM1 = SigX0 - CovX0M1*inv(SigM0)*CovM1X0

	# Compute mutual information I[M_{t-1}^{(r)}; X_{t}] for
	# each r.

	Is = numpy.zeros(len(partition))

	for partition_ind in range(len(partition)):
		partition_nodes = partition[partition_ind]

		block_inds = numpy.ix_(partition_nodes, partition_nodes)

		SigM0 = SigX0[block_inds]

		# What I had originally:

		block_inds = numpy.ix_(all_nodes, partition_nodes)

		# Cov(X_{t}, M^{(k)}_{t-1})

		CovX0M1 = CovX0X1[block_inds]

		# A possible alternative, that I currently (080720)
		# think is wrong:

		# block_inds = numpy.ix_(partition_nodes, all_nodes)
		# CovX0M1 = CovX0X1[block_inds].T

		# block_inds = numpy.ix_(all_nodes, partition_nodes)
		# C2 = CovX0X1[block_inds]

		# SigX0gM1 = SigX0 - CovX0M1*inv(SigM1)*CovM1X0
		# 
		# where SigM1 = SigM0 by wide-sense stationarity.

		SigX0gM1 = SigX0 - CovX0M1 @ numpy.linalg.solve(SigM0, CovX0M1.T)

		IX0M1 = 0.5*numpy.log(numpy.linalg.det(SigX0)/numpy.linalg.det(SigX0gM1))

		Is[partition_ind] = IX0M1

	psi = IX0X1 - numpy.max(Is)

	return psi

def part(agents, items):
	# From
	# 	https://stackoverflow.com/questions/42290859/generate-all-equal-sized-partitions-of-a-set

	if len(agents) == 1:
		yield {agents[0]: items}
	else:
		quota = len(items) // len(agents)
		for indexes in combinations(range(len(items)), quota):
			remainder = items[:]
			selection = [remainder.pop(i) for i in reversed(indexes)][::-1]
			for result in part(agents[1:], remainder):
				result[agents[0]] = selection
				yield result

def bipart(K):
	if K % 2 != 0:
		print("Error: This code has not been tested with uneven bipartitions.")

		return False
	else:
		partitions = list(part(["0", "1"], list(range(K))))

		return(partitions[:len(partitions)//2])