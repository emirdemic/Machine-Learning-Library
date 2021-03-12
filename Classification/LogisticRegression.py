import numpy as np

class LogisticRegression:
	def __init__(self, eta, n_passes, convergence_method = 'gradient_descent'):
		self.eta = eta
		self.n_passes = int(n_passes)
		if convergence_method not in ['gradient_descent', 'stochastic_gradient_descent', 'newton_raphson']:
			raise ValueError('convergence_method expected on of gradient_descent, stochastic_gradient_descent, \
				newton-raphson, but received:' + str(convergence_method))
		else:
			self.convergence_method = convergence_method


	def sigmoid(self, X, w, b):
		z = np.dot(X, w) + b
		activation = 1.0 / (1.0 + np.exp(np.negative(np.clip(z, -250, 250))))

		return activation


	def gradient_descent(self, X, y):
		activation = self.sigmoid(X, self.weights, self.intercept)
		dw = np.dot(X.T, (activation - y)) / X.shape[0]
		db = np.sum(activation - y) / X.shape[0]
		self.weights = self.weights + (self.eta * dw)
		self.intercept = self.intercept + (self.eta * db)


	def stochastic_gradient_descent(self, X, y):
		for xi, yi in zip(X, y):
			activation = self.sigmoid(xi, self.weights, self.intercept)
			dw = (activation - yi) * xi
			db = activation - yi
			self.weights = self.weights + (self.eta * dw)
			self.intercept = self.intercept + (self.eta * db)


	def newton_raphson(self, X, y): # TODO finish this function
		pass


	def fit(self, X, y):
		if self.convergence_method == 'gradient_descent':
			self.weights = np.random.randn(X.shape[1], 1)
			self.intercept = np.random.randn(1, 1)
			for one_pass in range(self.n_passes):
				self.gradient_descent(X, y)
				
		elif self.convergence_method == 'stochastic_gradient_descent':
			self.weights = np.random.randn(X.shape[1])
			self.intercept = np.random.randn(1)
			for one_pass in range(self.n_passes):
				self.stochastic_gradient_descent(X, y)

		elif self.convergence_method == 'newton_raphson':
			self.weights = np.random.randn(X.shape[1], 1)
			self.intercept = np.random.randn(1, 1)
			for one_pass in range(self.n_passes):
				self.newton_raphson(X, y)


	def predict(self, X):
		activation = self.sigmoid(X, self.weights, self.intercept)
		return activation


	def evaluate(self, X, y):
		activation = np.where(self.sigmoid(X, self.weights, self.intercept) >= 0.5, 1, 0)
		metrics = {}
		metrics['accuracy'] = np.sum(activation == y) / y.shape[0]
		return metrics