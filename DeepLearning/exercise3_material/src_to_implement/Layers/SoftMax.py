from Layers.Base import BaseLayer
import numpy as np


class SoftMax(BaseLayer):
    def __init__(self):
        super().__init__()
        self.trainable = False
        self.probs = None

    def forward(self, input_tensor):

        # Shift the input tensor to the negative domain
        shifted_input = input_tensor - np.max(input_tensor, axis=1, keepdims=True)
        # e** all elements in shifted_input
        exp_shifted_input = np.exp(shifted_input)
        # sum all elements of exp by axis  = 1
        exp_shifted_sum = np.sum(exp_shifted_input, axis=1, keepdims=True)
        # calculate softmax by dividing  input_tensor_exp/ input_tensor_sum - thank to numpy broadcasting
        self.probs = exp_shifted_input / exp_shifted_sum
        return self.probs

    def backward(self, error_tensor):
        batch_size = error_tensor.shape[0]
        input_tensor = np.zeros_like(error_tensor)
        for i in range(batch_size):
            jacobian = np.diag(self.probs[i]) - np.outer(self.probs[i], self.probs[i])
            input_tensor[i] = np.dot(error_tensor[i], jacobian)
        return input_tensor
