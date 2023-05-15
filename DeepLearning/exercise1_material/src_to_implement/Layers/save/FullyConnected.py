from Layers.Base import BaseLayer
from Optimization.Optimizers import Sgd
import numpy as np

class FullyConnected(BaseLayer):

    def __init__(self, input_size, output_size) -> None:
        super().__init__()
        self.trainable = True
        self.input_size = input_size
        self.output_size = output_size
        #random.uniform(low=0.0, high=1.0, size=None), default low =0 and  high = 1, 
        self.weights = np.random.uniform(0, 1, (self.input_size   , self.output_size))
        self.bias = np.random.uniform(0, 1, (1, self.output_size))
        
        #self.weights = np.random.uniform(size=(self.input_size + 1, self.output_size))
        #self.bias = np.random.uniform(size=self.output_size)
        self._optimizer = None
    
    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        #self.batch_size = input_tensor.shape[0]
        # Perform matrix multiplication between input tensor and weights
        output_tensor = np.dot(self.input_tensor , self.weights) + self.bias

        print(output_tensor.shape)
        
        
        # Apply activation function (e.g., ReLU)
        #output_tensor = np.maximum(0, output_tensor)
        return output_tensor

    # Since the default behavior of property is getter then: 
    @property 
    def optimizer(self):
        return self._optimizer
    # setter for optimizer
    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer
    
    def backward(self, error_tensor):
        # Compute the gradients of the weights and bias
        self._gradient_weights = np.dot(self.input_tensor.T, error_tensor)
        self._grad_bias = np.sum(error_tensor, axis=0)
        
       
        # Compute the error tensor for the previous layer
        #error_prev_layer = np.dot(error_tensor, self.weights.T)
        
        # Update the weights and bias using the optimizer
        if self._optimizer:
            self.weights = self._optimizer.calculate_update(self.weights, self.gradient_weights)
            self.bias = self._optimizer.calculate_update(self.bias, self._grad_bias)
        
        # Compute error tensor for previous layer
        prev_error_tensor = np.dot(error_tensor, self.weights.T)
        return prev_error_tensor
    
    @property 
    def gradient_weights(self):
        return self._gradient_weights