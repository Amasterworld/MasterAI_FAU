import numpy as np


class L2_Regularizer:

	def __init__(self, alpha):
		self.alpha = alpha

	def calculate_gradient(self, weights):
		# calculate sub-gradient ont he weights needed for optimizer
		return self.alpha * weights

	def norm(self, weights):
		# norm enhanced loss
		return self.alpha * np.sum((weights ** 2))


class L1_Regularizer:

	def __init__(self, alpha):
		self.alpha = alpha

	def calculate_gradient(self, weights):
		# calculate sub-gradient ont he weights needed for optimizer
		return self.alpha * (weights > 0) - self.alpha * (weights < 0)

	def norm(self, weights):
		return self.alpha * np.sum(np.abs(weights))
