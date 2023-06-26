from Layers.Base import BaseLayer
import numpy as np


class Flatten(BaseLayer):
    def __init__(self) -> None:
        super().__init__()
        self.trainable = False
        self.input_shape = None

    def forward(self, input_tensor):
        # save the shape of input tensor for the backward process
        self.input_shape = input_tensor.shape

        # Keep the batch size and flatten the data for exmaple images
        # for example if the input is 32, 28, 28, 3 - 32 is batch size ->keep it it is stored input_tensor.shape[0]
        # 28 are width and height of the images, 3 is color channel. -> 28*28*3
        output_tensor = np.reshape(input_tensor, (input_tensor.shape[0], (-1)))

        return output_tensor

    def backward(self, error_tensor):
        output_tensor = np.reshape(error_tensor, (self.input_shape))
        # note that you also can use: output_tensor = error_tensor.reshape(self.input_shape)
        return output_tensor