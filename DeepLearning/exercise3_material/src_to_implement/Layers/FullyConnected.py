from Layers.Base import BaseLayer
# from Optimization.Optimizers import Sgd
import numpy as np
import copy


class FullyConnected(BaseLayer):

	def __init__(self, input_size, output_size) -> None:
		super().__init__()
		self.trainable = True
		self.input_size = input_size
		self.output_size = output_size
		# random.uniform(low=0.0, high=1.0, size=None), default low =0 and  high = 1,
		self.weights = np.random.uniform(0, 1, (self.input_size + 1, self.output_size))
		self.bias = np.random.uniform(0, 1, (1, self.output_size))

		self._optimizer = None
		self._input_tensor = None



	def initialize(self, weights_initializer, bias_initializer):
		# note the weights_initializer can be an object of Xavier, He ... class
		# so it calls its initilalize
		self.weights = weights_initializer.initialize((self.input_size + 1, self.output_size), self.input_size,
		                                              self.output_size)

	# self.bias = bias_initializer.initialize((1, self.output_size), self.input_size, self.output_size)

	def forward(self, input_tensor):
		#print("self.weight shape ", self.weights.shape)
		bias = np.ones((input_tensor.shape[0], 1))
		input_tensor = np.hstack((input_tensor, bias))
		self._input_tensor = input_tensor.copy()
		#print("input_tensor shape ", self.input_tensor.shape)
		return input_tensor @ self.weights

	def backward(self, error_tensor):
		# Compute the gradients of the weights and bias

		self._gradient_weights = self._input_tensor.T @ error_tensor

		if self._optimizer:
			self.weights = self._optimizer.calculate_update(self.weights, self.gradient_weights)

		# self.bias = self._optimizer.calculate_update(self.bias, self._grad_bias)

		# Compute error tensor for previous layer

		prev_error_tensor = error_tensor @ self.weights.T[:, :-1]

		return prev_error_tensor

	@property
	def gradient_weights(self):
		return self._gradient_weights

	@property
	def optimizer(self):
		return self._optimizer

	# setter for optimizer
	@optimizer.setter
	def optimizer(self, optimizer):
		self._optimizer = optimizer