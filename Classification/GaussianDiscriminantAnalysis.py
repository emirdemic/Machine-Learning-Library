class GaussianDiscriminantAnalysis:
	def __init__(self, alpha = 0):
		# TODO you have a dimensions problem for xi and means
		# think about reshaping them/expanding dims or smth else
		if alpha < 0.0 or alpha > 1.0:
			raise ValueError('alpha value should be between 0 and 1')
		else:
			self.alpha = alpha


	def means(self, X, y):
		means = {}
		for value in np.unique(y):
			means[value] = np.sum(X[np.where(y == value)[0]], axis = 0) / np.where(y == value)[0].size

		self.means = means


	def phi(self, X, y):
		phi = {}
		for value in np.unique(y):
			phi[value] = np.sum(y == value) / X.shape[0]
		self.phi = phi


	def covariance_matrix(self, X):
		common_covariance = np.cov(X.T, rowvar = True)
		class_covariances = {}
		for value in np.unique(y):
			class_covariances[value] = self.alpha * np.cov(X[np.where(y == value)[0]].T, rowvar = True) + \
			 (1 - self.alpha) * common_covariance

		self.common_covariance = common_covariance
		self.class_covariances = class_covariances


	def gaussian(self, x, means, covariance):
		x = np.expand_dims(x, axis = 0)
		first_term = 1.0 / (((2.0 * np.pi)**(x.shape[0] / 2.0)) * np.sqrt(np.linalg.det(covariance)))
		second_term = np.exp(
			-0.5 * np.dot(
				(x - means), 
				np.dot(
					np.linalg.inv(covariance), 
					(x - means).T
					)
				)
			)

		return first_term * second_term


	def fit(self, X, y):
		self.means(X, y)
		self.phi(X, y)
		self.covariance_matrix(X)
		self.classes = np.unique(y)


	def predict(self, X):
		# finish this one as well and upload code to github
		# nesto nije dobro
		y = np.array([])
		for x in X:
			output_class = 0
			prob = 0
			for mean, phi, covariance, classes in zip(self.means, self.phi, self.class_covariances, self.classes):
				gaussian = self.gaussian(x, self.means[mean], self.class_covariances[covariance])
				print(gaussian * phi)
				if gaussian * phi > prob:
					prob = gaussian
					output_class = classes
			y = np.append(y, output_class)

		return y


	def evaluate(self, X, y):
		metrics = {}
		metrics['accuracy'] = np.sum(self.predict(X) == y) / X.shape[0]
		return metrics