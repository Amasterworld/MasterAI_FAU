import copy  # for deepcopy
import numpy as np



class NeuralNetwork:

    def __init__(self, optimizer, weights_initializer, bias_initializer):
        self.optimizer = optimizer
        self.loss = []
        self.layers = []
        self.data_layer = None
        self.loss_layer = None
        self._phase = False

        self.label_tensor = None
        self.input_tensor = None

        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer

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
            layer.initialize(self.weights_initializer, self.bias_initializer)
        self.layers.append(layer)

    def train(self, iterations):
        self._phase = True  # Set the phase to training
        for i in range(iterations):
            # Perform forward pass
            output = self.forward()
            # Calculate loss and store it
            self.loss.append(output)

            # Perform backward pass
            self.backward()

        return self.loss

    def test(self, input_tensor):
        self._phase = False  # Set the phase to testing
        output = input_tensor
        for layer in self.layers:
            output = layer.forward(output)

        return output

    def norm(self, weights) :
        for layer in self.layers:
            self.regularization_loss += layer.regularization_loss

    @property
    def phase(self):
        return self._phase

    @phase.setter
    def phase(self, val):
        self._phase = val
        #for layer in self.layers:
        #    layer.phase = val
