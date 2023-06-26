import numpy as np
from Layers.Base import BaseLayer
class TanH(BaseLayer):
    def __init__(self):
        super().__init__()
        self.trainable = False
        self.X = None
        self.Y = None
    def forward(self, input_tensor):
        #self.X = input_tensor.copy()
        output_tensor = np.tanh(input_tensor)
        self.Y = output_tensor.copy()
        return output_tensor

    def backward(self, error_tensor):
        derivative = 1 - (self.Y ** 2)
        output_error_tensor = error_tensor * derivative
        return output_error_tensor
