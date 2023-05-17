from Layers.Base import BaseLayer
#from Optimization.Optimizers import Sgd
#from Layers.FullyConnected import FullyConnected
import numpy as np


class ReLU(BaseLayer):
    def __init__(self):
        super().__init__()
        self.X = None
        self.trainable = False

    def forward(self, input_sensor):
        self.X = input_sensor.copy()  # Create a copy of the input tensor

        # Apply ReLU activation function element-wise
        output_tensor = np.maximum(input_sensor, 0)

        return output_tensor

    def backward(self, error_tensor):
        # Calculate the gradient using the ReLU derivative
        # remember that ReLU derivative = if x <= 0, = 1 if x > 0
        # then numpy where create a gradient array with the same X's shape
        # then use the mask to filter x, if x in X: > 0 ->1 in gradient, otherwise 0
        gradient = np.where(self.X > 0, 1, 0)  # Calculate the gradient using the ReLU derivative

        input_tensor = error_tensor * gradient  # Element-wise multiplication with the gradient

        return input_tensor
