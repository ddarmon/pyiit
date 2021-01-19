import pymvgc
import numpy

from itertools import combinations

def integrated_synergy(A, Sigma, partition, p=1):
	"""
	Compute the integrated synergy under the
	assumption that the underlying process is a Gaussian 
	VAR(p) process with model matrix A (K x K x p) and
	noise variance-covariance matrix Sigma (K x K).

	integrated_synergy() uses the full model matrix 
	A to compute the needed variance matrices, but
	only computes the integrated synergy using the 
	number of lagged time points given by p.

	Parameters
	----------
	A : numpy.array
			The K x K x p model matrix for the Var(p)
			process.
	Sigma : numpy.array
			The K x K noise variance-covariance matrix.
	partition : dict
			The desired partition of the system.
	p : int
			The number of lagged time points used to
			compute the integrated synergy.

	Returns
	-------
	psi : float
			The integrated synergy of the given
			partition of the VAR(p) model.

	Notes
	-----
	See 

	Pedro Mediano, Anil K. Seth, and Adam B. Barrett. "Measuring integrated information: Comparison of candidate measures in theory and simulation." Entropy 21.1 (2019): 17.

	for more details on computed integrated synergy.

	Examples
	--------
	>>> import module_name
	>>> # Demonstrate code here.

	"""

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

def minimize_integrated_synergy(A, Sigma, p=1, verbose=True):
	"""
	Minimize the integrated synergy across all-possible 
	bi-partitions of the system under the assumption
	that the underlying process is a Gaussian 
	VAR(p) process with model matrix A (K x K x p) and
	noise variance-covariance matrix Sigma (K x K).

	minimize_integrated_synergy() uses the full model matrix 
	A to compute the needed variance matrices, but
	only computes the integrated synergy using the 
	number of lagged time points given by p.

	Parameters
	----------
	A : numpy.array
			The K x K x p model matrix for the Var(p)
			process.
	Sigma : numpy.array
			The K x K noise variance-covariance matrix.
	p : int
			The number of lagged time points used to
			compute the integrated synergy.
	verbose : bool
			Whether (default) or not to print the optimal
			partition and its integrated synergy.

	Returns
	-------
	psi_opt : float
			The minimal integrated synergy across all
			of the bipartitions.
	best_partition : float
			The (possibly non-unique) bipartition giving
			the minimal integrated synergy.
	psi_by_partition : numpy.array
			The integrated synergies for each of the
			bipartitions.
	partitions : list
			All possible bipartitions of the state
			variables.

	Notes
	-----
	See 

	Pedro Mediano, Anil K. Seth, and Adam B. Barrett. "Measuring integrated information: Comparison of candidate measures in theory and simulation." Entropy 21.1 (2019): 17.

	for more details on computed integrated synergy.

	Examples
	--------
	>>> import module_name
	>>> # Demonstrate code here.

	"""

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
	"""
	Compute all possible paritions of items into agents.

	Note the partitions returned by part() include "duplicate"
	partitions, for example:

	{'1': [3, 4, 5], '0': [0, 1, 2]} and
	{'1': [0, 1, 2], '0': [3, 4, 5]}

	are both returned by part().

	Parameters
	----------
	agents : list
			The labels for the sets that make up the
			partition, e.g. ['0', '1'] for a bipartition.
	items : list
			The items to be partitioned.

	Returns
	-------
	result : list
			A list of dictionaries, where each element of 
			result contains the partition as a dictionary
			where the keys are the elements of agents and
			the values are the lists of the items in that
			set in the partition.

	Notes
	-----
	From
		https://stackoverflow.com/questions/42290859/generate-all-equal-sized-partitions-of-a-set

	Examples
	--------
	>>> import module_name
	>>> # Demonstrate code here.

	"""

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
	"""
	Return all unique bipartitions of K elements, up to the labels
	of the partitions.

	NOTE: This code will only work with bipartitions of sets with
	even cardinality.

	Parameters
	----------
	K : int
			The number of elements to partition via a bipartition.

	Returns
	-------
	partitions : list
			A list of dictionaries, where each element of 
			result contains the partition as a dictionary
			where the keys are the elements of agents and
			the values are the lists of the items in that
			set in the partition.

	Notes
	-----
	Any notes go here.

	Examples
	--------
	>>> import module_name
	>>> # Demonstrate code here.

	"""
	if K % 2 != 0:
		print("Error: This code has not been tested with uneven bipartitions.")

		return False
	else:
		partitions = list(part(["0", "1"], list(range(K))))

		return(partitions[:len(partitions)//2])