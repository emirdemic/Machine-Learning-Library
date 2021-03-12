import numpy as np 
import itertools


class Distances:
	def __init__(self, metric):
		self.metric = metric


	def calculate_a_b_c(self, X):
		a = np.sum(combination[0][np.where(combination[0] == 1)] == combination[1][np.where(combination[0] == 1)])
		b = np.sum(combination[0][np.where(combination[0] == 1)] == combination[1][np.where(combination[0] == 0)])
		c = np.sum(combination[0][np.where(combination[0] == 0)] == combination[1][np.where(combination[0] == 1)])

		return a, b, c


	def inverse_similarity(self, similarity):
		distance = 1 / (1 + similarity) #TODO think about this inverse
		return distance


	def jaccard_distance(self, a, b, c):
		distance = (b + c) / (a + b + c)
		return distance


	def hellinger_distance(self, a, b, c):
		denominator = np.sqrt((a + b) * (a + c))
		distance = 2 * np.sqrt(1 - (a / denominator))
		return distance


	def chord_distance(self, a, b, c):
		denominator = np.sqrt((a + b) * (a + c))
		distance = np.sqrt(2 * (1 - (a / denominator)))
		return distance


	def dice_distance(self, a, b, c):
		distance = (b + c) / (2*a + b + c)
		return distance 


	def gilbert_wells_distance(self, a, b, c, n): 
		similarity = np.log(a) - np.log(n) - np.log((a + b) / n) - np.log((a + c) / n)
		distance = self.inverse_similarity(similarity)
		return distance


	def ochlai(self, a, b, c):
		denominator = np.sqrt((a + b) * (a + c))
		similarity = a / denominator 
		distance = self.inverse_similarity(similarity)
		return distance 


	def forbesi(self, a, b, c, n):
		similarity = (n * a) / ((a + b) * (a + c))
		distance = self.inverse_similarity(similarity)
		return distance


	def sokal(self, a, b, c): 
		distance = (2 * (b + c)) / (a + 2 * (b + c))
		return distance


	def get_function(self, metric):
		if self.metric == 'jaccard':
			calculate_distance = self.jaccard_distance
		elif self.metric == 'hellinger':
			calculate_distance = self.hellinger_distance
		elif self.metric == 'chord':
			calculate_distance = self.chord_distance 
		elif self.metric == 'dice':
			calculate_distance = self.dice_distance 
		elif self.metric == 'gilbert_wells':
			calculate_distance = self.gilbert_wells_distance 
		elif self.metric == 'ochlai':
			calculate_distance = self.ochlai
		elif self.metric == 'forbesi':
			calculate_distance = self.forbesi 
		elif self.metric == 'sokal':
			calculate_distance = self.sokal

		return calculate_distance		


	def fit(self, X):
		calculate_distance = self.get_function(self.metric)

		dissimilarity_matrix = np.zeros((X.shape[1], X.shape[1]))
		row_index = 0
		column_index = 1 
		for combination in itertools.combinations(X.T, r = 2):
			a, b, c = self.calculate_a_b_c(X)
			dissimilarity_matrix[row_index, column_index] = calculate_distance(X, a, b, c)
			column_index += 1
			if column_index == dissimilarity_matrix.shape[1]:
				row_index += 1
				column_index = row_index + 1

		dissimilarity_matrix = dissimilarity_matrix + dissimilarity_matrix.T 

		return dissimilarity_matrix


def get_distances(X, metric):
	distance = Distances(metric)
	dissimilarity_matrix = distance.fit(X)
	return dissimilarity_matrix


class MultiDimensionalScaling:
	# TODO finish this, delete binary dissimilarity since you've written class distances
	def __init__(self, metric, n_dimensions):
		self.metric = metric 
		self.n_dimensions = n_dimensions


	def binary_dissimilarity(self, X, distance = 'jaccard', min_percent = False, weight = False): 
		# works like a charm :) 
		dissimilarity_matrix = np.zeros((X.shape[1], X.shape[1]))
		# btw find out why np.empty and np.full are idiotic ?
		row_index = 0
		column_index = 1 
		for combination in itertools.combinations(X.T, r = 2):
			b_c = np.sum(combination[0] != combination[1])
			a = np.sum(combination[0][np.where(combination[0] == 1)] == combination[1][np.where(combination[0] == 1)])
			if distance == 'sorensen':
				a = 2 * a
			elif distance == 'sokal':
				b_c = 2 * b_c
			elif distance == 'custom':
				pass # INSERT WEIGHT AND MIN PERCENT
			dissimilarity_matrix[row_index, column_index] = b_c / (a + b_c)
			column_index += 1
			if column_index == dissimilarity_matrix.shape[1]:
				row_index += 1
				column_index = row_index + 1

		dissimilarity_matrix = dissimilarity_matrix + dissimilarity_matrix.T 
		similarity_matrix = 1 - dissimilarity_matrix

		return dissimilarity_matrix, similarity_matrix


	def sammon_mapping(self, approximated_distances, actual_distances):
		row_index = 0
		column_index = 1 
		sammon = 0
		for combination in itertools.combinations(approximated_distances, r = 2):
			difference = actual_distances[row_index, column_index] - np.linalg.norm(combination[0] - combination[1])
			sammon += difference / actual_distances[row_index, column_index]

		pass


	def fit(self, X):
		if self.metric == 'jaccard':
			dissimilarity_matrix = self.binary_jaccard_dissimilarity(X)
		approximated = np.random.normal(size = (X.shape[1], self.n_dimensions))


		# INSERT MDS HERE :) 
		pass