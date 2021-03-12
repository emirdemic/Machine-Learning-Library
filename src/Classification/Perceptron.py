import numpy as np

class Perceptron:
	# TODO solve seed problem
	def __init__(self, eta, n_passes):
		self.eta = eta
		self.n_passes = int(n_passes)


	def linear_combination(self, X, w, b):
		z = np.dot(X, w) + b
		return z


	def fit(self, X, y):
		self.weights = np.random.randn(X.shape[1])
		self.intercept = np.random.randn(1)

		for one_pass in range(self.n_passes):
			for xi, yi in zip(X, y):
				z = np.where(self.linear_combination(xi, self.weights, self.intercept) >= 0, 1, 0)
				self.weights += self.eta * ((yi - z) * xi)
				self.intercept += self.eta * (yi - z)


	def predict(self, X):
		z = np.where(self.linear_combination(X, self.weights, self.intercept) >= 0, 1, 0)
		return z


	def evaluate(self, X, y):
		z = np.where(self.linear_combination(X, self.weights, self.intercept) >= 0, 1, 0)
		metrics = {}
		metrics['accuracy'] = (np.sum(z.reshape(z.shape[0], 1) == y)) / y.shape[0]
		return metrics