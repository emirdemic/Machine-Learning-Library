class HammingBinaryClustering:
	def __init__(self, n_clusters):
		self.n_clusters = n_clusters



	def hamming_distance(self, X):
		pass 


	def fit(self, X):
		random_assignment = np.random.randint(1, self.n_clusters + 1, size = X.shape[0])
		random_assignment = random_assignment.reshape(X.shape[0], 1)
		X = np.hstack((X, random_assignment))

		