import numpy as np
from Layers.Base import BaseLayer
class Sigmoid(BaseLayer):
    def __init__(self):
        super().__init__()
        self.trainable = False
        self.X = None
        self.Y = None


    def forward(self, input_tensor):
        #self.X = input_tensor.copy()
        output_tensor = 1 / (1 + np.exp(-input_tensor))
        self.Y = output_tensor.copy()
        return output_tensor

    def backward(self, error_tensor):

        derivative = self.Y * (1 - self.Y)
        error_tensor = error_tensor * derivative
        return error_tensor
