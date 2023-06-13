from Layers.Base import BaseLayer
import numpy as np


#mode valid padding = absence of padding

class Pooling(BaseLayer):
    def __init__(self, stride_shape, pooling_shape):
        
        super().__init__()
        self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape
        self.trainable = False

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        batch_size, num_channels, height, width = input_tensor.shape
        stride_height, stride_width = self.stride_shape
        pool_height, pool_width = self.pooling_shape

        # Calculate output dimensions
        out_height = (height - pool_height) // stride_height + 1
        out_width = (width - pool_width) // stride_width + 1

        # Initialize output tensor
        output_tensor = np.zeros((batch_size, num_channels, out_height, out_width))

        # Perform max pooling
        for b in range(batch_size):
            for c in range(num_channels):
                for h in range(out_height):
                    for w in range(out_width):
                        vert_start = h * stride_height
                        vert_end = vert_start + pool_height
                        horiz_start = w * stride_width
                        horiz_end = horiz_start + pool_width

                        pool_slice = input_tensor[b, c, vert_start:vert_end, horiz_start:horiz_end]
                        output_tensor[b, c, h, w] = np.max(pool_slice)

        return output_tensor
    

   
    def backward(self, error_tensor):
        batch_size, num_channels, out_height, out_width = error_tensor.shape
        stride_height, stride_width = self.stride_shape
        pool_height, pool_width = self.pooling_shape

        # Initialize error tensor for previous layer
        error_tensor_prev = np.zeros(self.input_tensor.shape)

        # Perform backward pass
        for b in range(batch_size):
            for c in range(num_channels):
                for h in range(out_height):
                    for w in range(out_width):
                        vert_start = h * stride_height
                        vert_end = vert_start + pool_height
                        horiz_start = w * stride_width
                        horiz_end = horiz_start + pool_width

                        pool_slice = self.input_tensor[b, c, vert_start:vert_end, horiz_start:horiz_end]
                        mask = (pool_slice == np.max(pool_slice))
                        error_tensor_prev[b, c, vert_start:vert_end, horiz_start:horiz_end] += mask * error_tensor[b, c, h, w]

        return error_tensor_prev
