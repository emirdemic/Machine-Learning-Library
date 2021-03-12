import numpy as np 
import itertools

# TODO documentation
# TODO worst_fit
# TODO category_utility_function

class COOLCAT:
	def __init__(self, n_clusters, batch_size, n_worst, initiate = True, confidence = False, m = False, sample_bound = False):
		self.n_clusters = int(n_clusters)
		self.batch_size = int(batch_size)
		self.n_worst = int(n_worst)
		self.confidence = confidence
		self.initiate = initiate
		self.sample_bound = sample_bound
		self.m = m


	def entropy(self, p):
		'''
		Calculates entropy of one variable. Vectorized.
		Args:
			p (np.array): variable
		Returns:
			e (float): entropy of variable
		'''

		prob = np.unique(p, return_counts = True)[1] / p.shape[0]
		e = np.negative(np.sum(prob * np.log2(prob)))

		return e


	def multivariate_entropy(self, X): 
		'''
		Calculates entropy of multiple variables with the assumption of 
		variable independence. In essence, E(X) = E(X1) + E(X2) + ... + E(Xp)
		Args:
			X (np.array): numpy matrix 
		Returns:
			multivariate_e (float): multivariate entropy

		'''

		multivariate_e = 0 # TODO vectorize; will save a tremendous amount of time
		for column in X.T:
			multivariate_e += self.entropy(column)

		return multivariate_e


	def system_entropy(self, X, n_clusters): 
		'''
		Calculates system entropy (i.e. expected entropy), which is defined as sum over 
		multivariate entropy of a cluster multiplied by the ratio of cluster cardinality 
		and dataset cardinality.
		Args:
			X (np.array): numpy matrix
			n_cluster (int): number of clusters
		Returns:
			system e (float): system (expected) entropy
		'''

		system_e = 0
		for cluster in range(1, n_clusters + 1):
			system_e += (X[X[:, -1] == cluster, :].shape[0] / X.shape[0]) * \
			(self.multivariate_entropy(
				X[X[:, -1] == cluster, :-1]
				)
			)

		return system_e


	def hamming_distance(self, X):
		'''
		Calculates Hamming distance (number of bit positions in which two bits are different).
		Identical to multivariate entropy of two objects with assumption of variable independence, 
		but computationally more efficient. Vectorized.
		Args:
			X (np.array): numpy matrix
		Returns:
			hamming (int): Hamming distance
		'''

		hamming = np.count_nonzero(X[0, :] != X[1, :])
		return hamming


	def chernoff_bounds(self, X, confidence, m):
		'''
		Calculates Chernoff bounds, i.e. the lower bound of the size of random sample drawn 
		from input data which guarantees with high probability the existence of at least one 
		member of each cluster, given the number of clusters.
		Args:
			X (np.array): numpy matrix
			confidence (float): probability of finding at least a member of each cluster
			m (int): size of the smallest cluster
		Returns:
			s (float): the lower boundary for the size of a random sample
		'''

		p = (X.shape[0] / self.n_clusters) / m
		s = self.n_clusters * p + self.n_clusters * p * np.log(1 / confidence) + \
			self.n_clusters * p * np.sqrt(np.square(np.log(1 / confidence)) + 2 * np.log(1 / confidence))

		return s		


	def worst_fit(self, X, size):

		cluster_lengths = {cluster: X[X[:, -1] == cluster, :].shape[0] for cluster in np.unique(X[:, -1])}
		X_temp = np.copy(X)
		X_temp = np.hstack((X_temp, np.ones((X_temp.shape[0], 1))))
		# TODO calculate probabilities for each cluster, store them into dictionary and then calculate p for datapoints
		# It will be much much faster
		for xi, cluster in zip(X_temp[X_temp[:, -2] != 0, :], X_temp[X_temp[:, -2] != 0, -2]):
			prob = np.product(
				np.apply_along_axis(
					np.count_nonzero, 0, X_temp[X_temp[:, -2] == cluster, :-2] == xi[:-2]
					) / cluster_lengths[cluster]
				) 
			xi[-1] = prob

		sorted_probs = np.argsort(X_temp[:, -1])
		X_temp[sorted_probs[: size], -2] = 0

		return X_temp[:, :-1]


	def initialization(self, X, confidence, m, sample_bound): 

		X = np.hstack((np.arange(0, X.shape[0]).reshape(X.shape[0], 1), X)) # adding index indicator variable
		X = np.hstack((X, np.zeros(X.shape[0]).reshape(X.shape[0], 1))) # adding cluster indicator variable 

		if sample_bound == True:
			sample_size = self.chernoff_bounds(X, confidence, m)
			sample = X[np.random.choice(
				np.arange(0, X.shape[0]), size = int(sample_size), replace = False
				), :]
		else:
			sample = X

		indexes = []
		while len(indexes) < self.n_clusters:
			temp_entropy = -1
			if len(indexes) == 0:
				for combination in itertools.combinations(sample, r = 2):
					pairwise_entropy = self.hamming_distance(np.array(combination)[:, 1:-1])
					if pairwise_entropy > temp_entropy:
						temp_entropy = pairwise_entropy
						indexes = np.array(combination)[:, 0]
				for index, cluster in zip(indexes, [1, 2]):
					sample[sample[:, 0] == index, -1] = cluster

			elif len(indexes) >= 2:
				temp_index = 0
				for xi in sample:
					xi = xi.reshape(1, xi.shape[0])
					pairwise_entropy = []
					for product in itertools.product(sample[sample[:, -1] != 0, :], xi): 
						pairwise_entropy.append(self.hamming_distance(np.array(product)[:, 1:-1]))
					if min(pairwise_entropy) > temp_entropy:
						temp_entropy = min(pairwise_entropy)
						temp_index = xi[0, 0]
				indexes = np.append(indexes, temp_index)
				sample[sample[:, 0] == indexes[-1], -1] = len(indexes)

			temp_entropy = 0

		for index, cluster in zip(indexes, np.arange(1, len(indexes) + 1)):
			X[X[:, 0] == index, -1] = cluster

		return X[:, 1:] 


	def fit(self, X):

		X = np.copy(X)
		if self.initiate == True:
			X = self.initialization(X, self.confidence, self.m, self.sample_bound)

		n_progress = 0

		while 0 in X[:, -1]:
			for xi in X: 
				if n_progress == self.batch_size:
					X = self.worst_fit(X, self.n_worst)
					n_progress = np.negative(self.n_worst)
					break
				else:
					if xi[-1] == 0:
						temp_sys_entropy = np.inf
						final_cluster = 1
						for cluster in range(1, self.n_clusters + 1):
							xi[-1] = cluster
							entropy = self.system_entropy(X[X[:, -1] != 0, :], self.n_clusters)
							if entropy < temp_sys_entropy:
								temp_sys_entropy = entropy
								final_cluster = cluster

						xi[-1] = final_cluster 

						n_progress += 1

		self.entropy_result = self.system_entropy(X, self.n_clusters)

		return X


	def category_utility_function(self, X):

		input_shape = X.shape[0]
		cluster_counts = np.unique(X[:, -1], return_counts = True)
		cluster_probs = {cluster: count / input_shape for cluster, count in zip(cluster_counts[0], cluster_counts[1])}


def run_clustering(df, n_clusters, batch_size, n_worst, initiate):
	clustering = COOLCAT(
		n_clusters = n_clusters,
		batch_size = batch_size,
		n_worst = n_worst,
		initiate = initiate,
		confidence = False,
		m = False, 
		sample_bound = False
		)
	clustered_df = clustering.fit(df)
	return clustered_df, clustering.entropy_result