"""
Batch Normalization: what it is ? and why do we need it?

Batch Normalization layer can be applied to both convolutional layers (which produce image-like tensors 4D) and fully connected layers (which produce vector-like tensors 2D).
"""

from Layers.Base import BaseLayer
import numpy as np


class BatchNormalization(BaseLayer):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.trainable = True
        self.weights = None
        self.bias = None
        self.running_mean = None
        self.running_var = None
        self.eps = 1e-10
        self.testing_phase = False
        self.input_tensor_shape = None
        self.input_tensor = None

        # for test update
        self._optimizer = None

    def initialize(self):
        if self.channels is None:
            self.weights = np.ones(())
            self.bias = np.zeros(())
        else:
            self.weights = np.ones(self.channels)
            self.bias = np.zeros(self.channels)

    def forward(self, input_tensor):

        #self.input_tensor = input_tensor
        if input_tensor.ndim == 4:  # Image-like tensor

            self.input_tensor_shape = input_tensor.shape
            #batch_size, channels, height, width  = input_tensor.shape
            reformat_input_tensor = self.reformat(input_tensor)  # Convert to vector-like tensor


        else:
            reformat_input_tensor = input_tensor
        self.input_tensor = reformat_input_tensor


        if self.weights is None or self.bias is None:  # Initialize gamma and beta if not already initialized
            self.initialize()

        mean = np.mean(reformat_input_tensor, axis=0)
        var = np.var(reformat_input_tensor, axis=0)

        # if testing phase is True then assign mean and var to self.running_mean and self.running_var
        #  self.running_mean and self.running_var are calculated in the training phase
        # Therefore in the testing phease mean and var are constant
        if self.testing_phase:
            # Use running mean and variance during testing phase
            mean = self.running_mean
            var = self.running_var
        else:
            # Update running mean and variance using moving average estimation
            momentum = 0.8
        # in the training phase update self.running_mean and self.running_var
            if self.running_mean is None:
                self.running_mean = mean
            else:
                self.running_mean = momentum * self.running_mean + (1 - momentum) * mean

            if self.running_var is None:
                self.running_var = var
            else:
                self.running_var = momentum * self.running_var + (1 - momentum) * var

        normalized = (reformat_input_tensor - mean) / np.sqrt(var + self.eps)
        output_tensor = self.weights * normalized + self.bias

        if input_tensor.ndim == 4:  # Convert back to image-like tensor
            output_tensor = self.reformat(output_tensor)

        return output_tensor

    def backward(self, error_tensor):

        error_tensor_shape = error_tensor.shape
        # Reshape error tensor if necessary
        if error_tensor.ndim == 4:
            self.input_tensor_shape = error_tensor.shape

            error_tensor = self.reformat(error_tensor)

        # Compute intermediate values
        mean = np.mean(self.input_tensor, axis=0)
        var = np.var(self.input_tensor, axis=0)
        std_inv = 1.0 / np.sqrt(var + self.eps)
        normalized = (self.input_tensor - mean) * std_inv

        m = error_tensor.shape[0] # batch* height* width?

        # Compute gradients w.r.t. gamma and beta
        self.gradient_weights = np.sum(error_tensor * normalized, axis=0)
        self.gradient_bias = np.sum(error_tensor, axis=0)

        if self._optimizer:
            self.weights = self._optimizer.calculate_update(self.weights, self.gradient_weights)

        # Compute gradient w.r.t. input tensor
        d_normalized = error_tensor * self.weights
        d_var = np.sum(d_normalized * (self.input_tensor - mean) * -0.5 * std_inv ** 3, axis=0)
        d_mean = np.sum(d_normalized * -std_inv, axis=0) + d_var * np.mean(-2.0 * (self.input_tensor - mean), axis=0)
        input_gradient = d_normalized * std_inv + d_var * 2.0 * (self.input_tensor - mean) / m + d_mean / m

        # Reshape input gradient if necessary
        if len(error_tensor_shape) == 4:
            input_gradient = self.reformat(input_gradient)


        return input_gradient

    def reformat(self, tensor):
        if tensor.ndim == 4:  # Image-like tensor
            batch_size, channels , height, width  = tensor.shape

            tensor = tensor.transpose((0, 2, 3, 1)).reshape(batch_size * height * width, channels)  # Reshape to B · M · N × H
        elif tensor.ndim == 2:  # Vector-like tensor
            # Calculate the height and width dimensions based on the number of channels

            tensor = tensor.reshape(self.input_tensor_shape[0], self.input_tensor_shape[2],  self.input_tensor_shape[3], self.input_tensor_shape[1]).transpose(0, 3, 1, 2)  # Reshape to B × H × M × N
        else:
            raise ValueError("Invalid tensor shape")

        return tensor

    # for test update
    @property
    def optimizer(self):
        return self._optimizer

    # setter for optimizer
    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer