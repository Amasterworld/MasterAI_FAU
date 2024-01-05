from Optimization.Optimizers import Sgd

import copy  # for deepcopy
import numpy as np
from Layers.SoftMax import SoftMax


class NeuralNetwork:

    def __init__(self, optimizer, weights_initializer, bias_initializer):
        self.optimizer = optimizer
        self.loss = []
        self.layers = []
        self.data_layer = None
        self.loss_layer = None

        self.label_tensor = None
        self.input_tensor = None

        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer

    def forward(self):

        self.input_tensor, self.label_tensor = self.data_layer.next()
        # 
        output_tensor = self.input_tensor
        # traverse the layer in the given layers, note that layers are added to self.layers by method: append_layer
        for layer in self.layers:
            # the output_tensor of the current layer is the input of the next layer.
            output_tensor = layer.forward(output_tensor)
         # calculate the loss by using self.label_tensor - the output of the last layers in the forward process
        loss_output = self.loss_layer.forward(output_tensor, self.label_tensor)
        return loss_output

    def backward(self):
        # the initial error_tensor is calculating  by self.label_tensor and the last output_tensor from the forward process
        # See the Loss layer
        error_tensor = self.loss_layer.backward(self.label_tensor)
        for layer in reversed(self.layers):
            error_tensor = layer.backward(error_tensor)

    def append_layer(self, layer):
        if layer.trainable:
            layer.optimizer = copy.deepcopy(self.optimizer)
            layer.initialize(self.weights_initializer, self.bias_initializer)
        self.layers.append(layer)

    def train(self, iterations):
        for i in range(iterations):
            # Perform forward pass
            output = self.forward()
            # Calculate loss and store it
            self.loss.append(output)

            # Perform backward pass
            self.backward()

        return self.loss

    def test(self, input_tensor):
        output = input_tensor
        for layer in self.layers:
            output = layer.forward(output)

        return output
