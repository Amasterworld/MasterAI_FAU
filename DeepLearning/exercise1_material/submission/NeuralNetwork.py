from Optimization.Optimizers import Sgd

import copy  # for deepcopy
import numpy as np
from Layers.SoftMax import SoftMax


class NeuralNetwork:

    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.loss = []
        self.layers = []
        self.data_layer = None
        self.loss_layer = None

        self.label_tensor = None
        self.input_tensor = None

    def forward(self):

        self.input_tensor, self.label_tensor = self.data_layer.next()
        output_tensor = self.input_tensor
        for layer in self.layers:
            output_tensor = layer.forward(output_tensor)
        loss_output = self.loss_layer.forward(output_tensor, self.label_tensor)
        return loss_output

    def backward(self):
        # label_tensor = self.label_tensor
        error_tensor = self.loss_layer.backward(self.label_tensor)
        for layer in reversed(self.layers):
            error_tensor = layer.backward(error_tensor)

    def append_layer(self, layer):
        if layer.trainable:
            layer.optimizer = copy.deepcopy(self.optimizer)
        self.layers.append(layer)

    def train(self, iterations):
        for _ in range(iterations):
            # Perform forward pass
            output = self.forward()
            # Calculate loss and store it
            #current_loss = self.loss_layer.forward(output, self.label_tensor)
            self.loss.append(output)

            # Perform backward pass
            label_tensor = self.label_tensor
            self.backward()

            # Update weights and biases
            for layer in self.layers:
                if layer.trainable:
                    layer.weights = self.optimizer.calculate_update(layer.weights, layer.gradient_weights)
                    layer.bias = self.optimizer.calculate_update(layer.bias, layer._grad_bias)

        return self.loss

    def test(self, input_tensor):
        output = input_tensor
        for layer in self.layers:
            output = layer.forward(output)

        return output
