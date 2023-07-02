import copy

# Import numpy for matrix operations
import numpy as np
from Layers.Base import BaseLayer
from Layers.FullyConnected import FullyConnected
from Layers.Sigmoid import Sigmoid
from Layers.TanH import TanH


# Define the RNN class
class RNN(BaseLayer):

    # Constructor
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.trainable = True
        # Initialize the input, hidden and output sizes
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Initialize the hidden state with all zeros
        #self.hidden = np.zeros((hidden_size, 1))
        self.hidden = np.zeros((1, hidden_size))
        # Initialize the memorize property with False
        self.memorize = False

        self._weights = None
        # Initialize the weights and biases randomly
        self.Wxh = np.random.randn(hidden_size, input_size)  # Input to hidden weights
        self.Whh = np.random.randn(hidden_size, hidden_size)  # Hidden to hidden weights
        self.Why = np.random.randn(output_size, hidden_size)  # Hidden to output weights
        self.bh = np.random.randn(hidden_size, 1)  # Hidden bias
        self.by = np.random.randn(output_size, 1)  # Output bias

        self.fc1 = FullyConnected(self.input_size + self.hidden_size, self.hidden_size)
        self.fc2 = FullyConnected(self.hidden_size, self.output_size)
        self.tanh_activation = TanH()
        self.sigmoid_activation = Sigmoid()

        # self.fc1_mem = []
        # self.fc2_mem = []
        # self.tanh_mem = []
        # self.sigmoid_mem = []
        # self.hidden_mem = []
        # self.fc1_grad = np.zeros_like(self.fc1.weights)
        # self.fc2_grad = np.zeros_like(self.fc2.weights)
        self.fc1_mem = []
        self.fc2_mem = []
        self.tanh_mem = []
        self.sigmoid_mem = []

        self._optimizer = None
        self._optimizer1 = None
        self._optimizer2 = None

        self._gradient_weights = None
    # Forward method
    def forward(self, input_tensor):
        self.fc1_mem = []
        self.fc2_mem = []
        self.tanh_mem = []
        self.sigmoid_mem = []
        # Get the sequence length (time dimension) from the input tensor
        seq_length = input_tensor.shape[0]
        output_tensor = np.zeros((seq_length, self.output_size))
        if not self.memorize:
            self.hidden = np.zeros((1, self.hidden_size))

        # Loop over the sequence
        for t in range(seq_length):
            x = input_tensor[t][None, :]
            # x = input_tensor[t].reshape(1, -1)
            # h = self.hidden[t][None, :]
            xh = np.concatenate((x, self.hidden), axis=1)

            fc1 = self.fc1.forward(xh)
            self.fc1_mem.append(self.fc1._input_tensor)

            self.hidden = self.tanh_activation.forward(fc1)
            self.tanh_mem.append(self.tanh_activation.Y)

            fc2 = self.fc2.forward(self.hidden)
            self.fc2_mem.append(self.fc2._input_tensor)
            # Compute the output vector using the hidden state
            y = self.sigmoid_activation.forward(fc2)
            self.sigmoid_mem.append(self.sigmoid_activation.Y)

            # Store the output vector in the output tensor
            output_tensor[t] = y

        # Return the output tensor

        return output_tensor

    # Backward method
    def backward(self, error_tensor):

        # Get the sequence length (time dimension) from the input tensor
        seq_length = error_tensor.shape[0]
        fc1_grad = np.zeros_like(self.fc1.weights)
        fc2_grad = np.zeros_like(self.fc2.weights)
        # Initialize the error tensor for the previous layer
        prev_error_tensor = np.zeros((seq_length, self.input_size))
        hidden_E = np.zeros((1, self.hidden_size))

        # Loop over the sequence in reverse order
        for t in reversed(range(seq_length)):
            # Get the error vector at time t
            error = error_tensor[t][None, :]
            # error = error_tensor[t].reshape(-1, 1)
            self.sigmoid_activation.Y = self.sigmoid_mem[t]
            sig_error = self.sigmoid_activation.backward(error)

            # Backpropagate through the second fully connected layer
            self.fc2._input_tensor = self.fc2_mem[t]
            fc2_error = self.fc2.backward(sig_error)
            fc2_grad += self.fc2.gradient_weights
            grad_hy = hidden_E + fc2_error

            # Backpropagate through the tanh activation function
            self.tanh_activation.Y = self.tanh_mem[t]
            tanh_error = self.tanh_activation.backward(grad_hy)

            # Backpropagate through the first fully connected layer
            self.fc1._input_tensor = self.fc1_mem[t]
            conca1_gradient = self.fc1.backward(tanh_error)
            fc1_grad += self.fc1.gradient_weights
            # print("conca1_gradient", conca1_gradient.shape)

            # x_gradient, hidden_E = np.split(conca1_gradient, [self.input_size], axis=1)
            x_gradient = conca1_gradient[:, :self.input_size]
            hidden_gradient = conca1_gradient[:, self.input_size:]
            hidden_E = hidden_gradient

            prev_error_tensor[t] = x_gradient.flatten()

            self.gradient_weights = fc1_grad
        if self._optimizer:
            self.fc1.weights = self._optimizer1.calculate_update(self.fc1.weights, fc1_grad)
            self.fc2.weights = self._optimizer2.calculate_update(self.fc2.weights, fc2_grad)

        # prev_error_tensor = np.flip(prev_error_tensor, axis=0)

        return prev_error_tensor

    def initialize(self, weights_initializer, bias_initializer):
        fan_in = self.input_size + self.hidden_size
        fan_out = self.hidden_size

        weight_shape = (fan_in + 1, fan_out)

        self.weights = weights_initializer.initialize(weight_shape, fan_in, fan_out)
        self.by = bias_initializer.initialize(self.bh.shape, fan_in, fan_out)

    @property
    def weights(self):
        return self.fc1.weights

    @weights.setter
    def weights(self, value):
        self.fc1.weights = value

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, value):
        self._gradient_weights = value

    @property
    def memorize(self):
        return self._memorize

    @memorize.setter
    def memorize(self, value):
        self._memorize = value

    @property
    def optimizer(self):
        return self._optimizer

    # setter for optimizer
    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer
        self._optimizer1 = copy.deepcopy(optimizer)
        self._optimizer2 = copy.deepcopy(optimizer)

    def calculate_regularization_loss(self):
        # get the regularization term from the optimizer
        regularization_term = self.optimizer.get_regularization_term(self.fc1.weights)
        # return the regularization loss as half of the regularization term
        return 0.5 * regularization_term
