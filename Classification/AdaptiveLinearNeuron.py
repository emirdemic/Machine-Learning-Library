class AdaptiveLinearNeuron:
	def __init__(self, eta, n_passes):
		self.eta = eta
		self.n_passes = int(n_passes)


	def linear_combination(self, X, w, b):
		z = np.dot(X, w) + b
		return z


	def fit(self, X, y):
		self.weights = np.random.randn(X.shape[1], 1)
		self.intercept = np.random.randn(1, 1)

		for one_pass in range(n_passes):
			activation = self.linear_combination(X, self.weights, self.intercept)
			error = activation - y
			self.weights = self.weights - (self.eta * np.dot(X.T, error))
			self.intercept = self.intercept - (self.eta * np.sum(error))


	def predict(self, X):
		z = np.where(linear_combination(X, self.weights, self.intercept) >= 0, 1, 0)
		return z


	def evaluate(self, X, y):
		z = np.where(self.linear_combination(X, self.weights, self.intercept) >= 0, 1, 0)
		metrics = {}
		metrics['accuracy'] = np.sum(z == y) / y.shape[0]