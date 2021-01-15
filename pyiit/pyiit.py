import pymvgc
import numpy

from itertools import combinations

def integrated_synergy(A, Sigma, partition, p=1):
	K = A.shape[0]

	all_nodes = list(range(K))

	#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	#
	# Get variance-covariance matrices:
	#
	#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	Gam, Gamma0 = pymvgc.var_to_autocov(A, Sigma, nlag=p+1, return_big_gamma=True)

	# NOTE: Gamma0 is equivalent to Var(X_{(t-1):(t-p)})

	# Var(X_{t})

	SigX0  = Gam[:, :, 0]

	# Cov(X_{t}, X_{(t - 1):(t-p)})

	CovX0X1 = Gam[:, :, 1:]

	# Reshape to be K x K*p, rather than K x K x p

	CovX0X1 = CovX0X1.reshape(K, K*p, order = 'F')

	#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	#
	# Compute I[X_{t-1}; X_t]
	#
	#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	# I[X_{t-1}; X_t] = 0.5 log |SigX0|/|SigX0gX1, return_big_Gamma=True|

	# SigX0gX1 = SigX0 - CovX0X1*inv(SigX1)*CovX1X0

	# Note: Computing inv(SigX1)*CovX1X0 by solving
	# 
	# 	SigX0*A = CovX1X0

	SigX0gX1 = SigX0 - CovX0X1 @ numpy.linalg.solve(Gamma0, CovX0X1.T)

	IX0X1 = 0.5*numpy.log(numpy.linalg.det(SigX0)/numpy.linalg.det(SigX0gX1))

	#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	#
	# Compute I[M_{t-1}^{(r)}; X_{t}] for each r.
	#
	#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	# I[M_{t-1}^{(r)}; X_{t}] = 0.5 log |SigX0|/|SigX0gM1|

	# SigX0gM1 = SigX0 - CovX0M1*inv(SigM1)*CovM1X0

	# Compute mutual information I[M_{t-1}^{(r)}; X_{t}] for
	# each r.

	Is = numpy.zeros(len(partition))

	for partition_ind in range(len(partition)):
		partition_nodes = partition[partition_ind]

		# Extract the appropriate rows and columns from
		# Gamma0 to pull out Var(M^{(k)}_{(t-p)x(t-1)}).

		# Ex. When p = 3, K = 6, and partition_nodes = [0, 1, 2]
		# should use rows and columns:
		# 
		# 0, 1, 2 | 3, 4, 5 | 6, 7, 8

		subset_inds = []

		for node in partition_nodes:
			subset_inds += [p*node + i for i in range(p)]

		block_inds = numpy.ix_(subset_inds, subset_inds)

		SigM1 = Gamma0[block_inds]

		block_inds = numpy.ix_(all_nodes, subset_inds)

		# Cov(X_{t}, M^{(k)}_{(t-p)x(t-1)})

		CovX0M1 = CovX0X1[block_inds]

		# SigX0gM1 = SigX0 - CovX0M1*inv(SigM1)*CovM1X0

		SigX0gM1 = SigX0 - CovX0M1 @ numpy.linalg.solve(SigM1, CovX0M1.T)

		IX0M1 = 0.5*(numpy.log(numpy.linalg.det(SigX0)) - numpy.log(numpy.linalg.det(SigX0gM1)))

		Is[partition_ind] = IX0M1

	psi = IX0X1 - numpy.max(Is)

	return psi

def maximize_integrated_synergy(A, Sigma, p=1, verbose=True):
	K = A.shape[0]

	partitions = bipart(K)

	### Optimal partition by iteration over all partitions:

	psi_by_partition = numpy.zeros(len(partitions))

	for partition_ind in range(len(partitions)):
		partition = [partitions[partition_ind]['0'], partitions[partition_ind]['1']]

		psi = integrated_synergy(A, Sigma, partition=partition, p=1)

		psi_by_partition[partition_ind] = psi

	best_partition = numpy.argmin(psi_by_partition)

	psi_opt = psi_by_partition[best_partition]

	if verbose:
		print("The optimal partition is: ", partitions[best_partition])
		print("The IIT is: ", psi_opt)

	return psi_opt, best_partition, psi_by_partition, partitions


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