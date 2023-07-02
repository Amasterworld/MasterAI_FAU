"""
Dropout: regularization technique,  It involves randomly deactivating or "dropping out" a fraction of the neurons in a layer during the training phase.
During dropout, each neuron has a probability (typically 0.5) of being temporarily ignored or "dropped out" for a particular input sample.
This means that its output value is set to zero, and it doesn't contribute to the forward pass of the network.

The primary purpose of dropout is to prevent overfitting, which occurs when a model becomes too specialized and performs poorly on unseen data.
By randomly dropping out neurons, dropout reduces the interdependencies between neurons and encourages the network to learn more robust and generalized features.
"""
import numpy as np
from Layers.Base import BaseLayer


class Dropout(BaseLayer):

    def __init__(self, probability):
        super().__init__()
        self.probability = probability
        self.trainable = False
        self.testing_phase = False
        self.mask = None

    def forward(self, input_tensor):
        if self.testing_phase:

            output_tensor = input_tensor
        else:
            # create mask, if the random  > the given probability then 1 otherwise 0
            self.mask = np.random.rand(*input_tensor.shape) < self.probability
            output_tensor = input_tensor * self.mask * 1 / self.probability

        return output_tensor

    def backward(self, error_tensor):
        # if in training phase, propagate the error only through the active units
        if self.testing_phase:

            output_error = error_tensor
        else:
            output_error = error_tensor * self.mask * 1 / self.probability
        return output_error
