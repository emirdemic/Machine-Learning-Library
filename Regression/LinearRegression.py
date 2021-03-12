import numpy as np

class LinearRegression:
	# TODO insert seed
	def __init__(self, convergence_method = 'normal_equation', eta = 0.01, n_passes = 3000):
		if convergence_method not in ['normal_equation', 'gradient_descent', 'stochastic_gradient_descent']:
			raise ValueError('convergence_method expected one of normal_equaton, gradient_descent, stochastic_gradient_descent, but received: ' \
				+ str(convergence_method))
		else:
			self.convergence_method = convergence_method
		self.eta = eta
		self.n_passes = int(n_passes)


	def linear_combination(self, X, w, b):
		z = np.dot(X, w) + b
		return z


	def normal_equation(self, X, y):
		X = np.hstack((np.ones((X.shape[0], 1)), X))
		z = np.linalg.inv(np.matmul(X.T, X))
		params = np.dot(z, np.matmul(X.T, y))
		self.intercept = params[0]
		self.weights = params[1: ]
		

	def transform(self, X, degree):
		for column in X.T:
			for power in range(2, degree + 1):
				new_column = np.power(column, power)
				new_column = np.expand_dims(new_column, axis = 1)
				X = np.hstack((X, new_column))

		return X


	def gradient_descent(self, X, y):
		activation = self.linear_combination(X, self.weights, self.intercept)
		error = activation - y
		dw = (np.dot(X.T, error)) / X.shape[0]

		self.weights = self.weights - (self.eta * dw)
		self.intercept = self.intercept - (self.eta * (np.sum(error) / X.shape[0]))


	def stochastic_gradient_descent(self, X, y):
		for xi, yi in zip(X, y):
			activation = self.linear_combination(xi, self.weights, self.intercept)
			error = activation - yi
			dw = error * xi
			self.weights = self.weights - (self.eta * dw)
			self.intercept = self.intercept - (self.eta * error)


	def fit(self, X, y):
		if self.convergence_method == 'normal_equation':
			self.normal_equation(X, y)

		elif self.convergence_method == 'gradient_descent':
			self.weights = np.random.randn(X.shape[1], 1).astype(dtype = np.float64)
			self.intercept = np.random.randn(1, 1).astype(dtype = np.float64)
			for one_pass in range(self.n_passes):
				self.gradient_descent(X, y)

		elif self.convergence_method == 'stochastic_gradient_descent':
			self.weights = np.random.randn(X.shape[1]).astype(dtype = np.float64)
			self.intercept = np.random.randn(1).astype(dtype = np.float64)
			for one_pass in range(self.n_passes):
				self.stochastic_gradient_descent(X, y)
			self.weights = np.expand_dims(self.weights, axis = 1)
			self.intercept = np.expand_dims(self.intercept, axis = 1)


	def predict(self, X):
		activation = self.linear_combination(X, self.weights, self.intercept)
		return activation


	def evaluate(self, X, y):
		sse = np.sum(
			np.square(
				(y - self.linear_combination(X, self.weights, self.intercept)
					)
				)
			)
		mse = sse / X.shape[0]
		rmse = np.sqrt(mse)
		sae = np.sum(
			np.abs(
				(y - self.linear_combination(X, self.weights, self.intercept)
					)
				)
			)
		mae = sae / X.shape[0]
		sst = np.sum(np.square(y - np.mean(y)))
		r2 = 1 - (sse / sst)
		
		metrics = {
			'sse' : sse, 
			'mse' : mse, 
			'rmse' : rmse, 
			'mae' : mae, 
			'r2' : r2
			}
			
		return metrics