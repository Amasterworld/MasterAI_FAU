import numpy as np


# from Layers.SoftMax import SoftMax


class CrossEntropyLoss:
    def __init__(self):
        self.predict = None

    def forward(self, predict_tensor, label_tensor):
        epsilon = np.finfo(float).eps  # Get the machine epsilon
        #store that last output_tensor of the forward process, that is used for the initial error_tensor in backward process
        self.predict = predict_tensor
        
        # Apply the formula: loss = -sum(label_tensor * log(predict_tensor + eps))
        loss = -np.sum(label_tensor * np.log(predict_tensor + epsilon))

        return loss

    def backward(self, label_tensor):
        
        return -label_tensor / self.predict
