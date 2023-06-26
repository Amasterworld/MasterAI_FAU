from Layers.Base import BaseLayer
import numpy as np
# from scipy.ndimage import correlate, convolve
from scipy.signal import correlate, convolve
import math
"""
The code lines have many rooms to improve.
 at least make a func: def padding(X, pad) for both forward and backward
"""

class Conv(BaseLayer):
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        super().__init__()
        self.stride_shape = stride_shape
        self.convolution_shape = convolution_shape
        self.num_kernels = num_kernels
        self.trainable = True

        # Initialize weights and bias
        self.weights = np.random.uniform(size=(self.num_kernels, *self.convolution_shape))
        self.bias = np.random.uniform(size=(self.num_kernels))
        self.input_tensor = None

        self.dW = None
        self.db = None

        self._optimizer = None

    def initialize(self, weights_initializer, bias_initializer):

        fan_in = np.prod(self.convolution_shape)

        fan_out = np.prod(self.convolution_shape[1:]) * self.num_kernels

        self.weights = weights_initializer.initialize(self.weights.shape, fan_in, fan_out)

        self.bias = bias_initializer.initialize(self.bias.shape, fan_in, fan_out)

    def conv_single_step(self, a_slice_prev, W, b):



        ### START CODE HERE ### (≈ 2 lines of code)
        # Element-wise product between a_slice and W. Do not add the bias yet.
        s = np.multiply(a_slice_prev, W)
        # Sum over all entries of the volume s.
        Z = np.sum(s)
        # Add bias b to Z. Cast b to a float() so that Z results in a scalar value.
        # print("Z = ", Z.shape)
        Z = Z + b
        # print("Z after plussing b ", Z.shape)
        ### END CODE HERE ###

        return Z

    def forward(self, input_tensor):
        """
        Implements the forward propagation for a convolution function

        Arguments:
        A_prev, input_tensor -- output activations of the previous layer, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
        W -- Weights-> seflf.weights, numpy array of shape ( n_C, n_C_prev, conv_h, conv_w)
        b -- Biases, numpy array of shape (as the size of kernel)

        Returns:
        Z -- conv output, numpy array of shape (m, n_C, n_H, n_W), note n_W can be missed if the input tensor contains only m, c, y
        self.input_tensor -- cache of values needed for the conv_backward() function
        """
        self.input_tensor = input_tensor
        ### START CODE HERE ###
        # Retrieve dimensions from input_tensor's shape (≈1 line)
        if len(input_tensor.shape) == 4:
            m, n_C_prev, n_H_prev, n_W_prev = input_tensor.shape
        else:
            m, n_C_prev, n_H_prev = input_tensor.shape

        if len(self.weights.shape) == 4:
            n_C, n_C_prev, conv_h, conv_w, = self.weights.shape  # number of kernel, c, m, n

            # . claculate pad width and left when it is 2D
            if conv_w % 2 == 0:
                self.pad_width_left = math.floor((conv_w - 1) / 2)
                self.pad_width_right = math.ceil((conv_w - 1) / 2)

            else:
                self.pad_width_left = self.pad_width_right = (conv_w - 1) // 2
        else:
            n_C, n_C_prev, conv_h = self.weights.shape  # number of kernel, c, m

        # what if conv_w % 2 != 0?
        if conv_h % 2 == 0:

            self.pad_height_top = math.floor((conv_h - 1) / 2)
            self.pad_height_bottom = math.ceil((conv_h - 1) / 2)

        else:
            self.pad_height_top = self.pad_height_bottom = (conv_h - 1) // 2

            # Compute the dimensions of the CONV output volume using the formula given above.

        if len(input_tensor.shape) == 4:  # input tensor contains:  conv_h and conv_w

            n_H = int((n_H_prev - conv_h + (self.pad_height_top + self.pad_height_bottom)) / self.stride_shape[0]) + 1
            n_W = int((n_W_prev - conv_w + (self.pad_width_left + self.pad_width_right)) / self.stride_shape[1]) + 1

        else:
            n_H = int((n_H_prev - conv_h + (self.pad_height_top + self.pad_height_bottom)) / self.stride_shape[0]) + 1
            n_W = 1
        # Initialize the output volume Z with zeros. (≈1 line)
        if len(input_tensor.shape) == 4:
            Z = np.zeros((m, n_C, n_H, n_W))
        else:
            Z = np.zeros((m, n_C, n_H))

        if len(input_tensor.shape) == 3:
            self.pad_width = ((0, 0), (0, 0), (self.pad_height_top, self.pad_height_bottom))
        else:  # Assuming the input_tensor.shape is (m, n_H_prev, n_W_prev, n_C_prev) for 4D tensors
            self.pad_width = (
            (0, 0), (0, 0), (self.pad_height_top, self.pad_height_bottom), (self.pad_width_left, self.pad_width_right))

        # Create input_tensor pad by padding input tensor
        A_prev_pad = np.pad(input_tensor, self.pad_width, mode='constant', constant_values=0)

        for i in range(m):
            a_prev_pad = A_prev_pad[i]
            for c in range(n_C):

                for h in range(n_H):
                    # input tensor is convolution_shape has shape = 3 (contain: c, conv_h, conv_w)
                    if n_W > 1:
                        for w in range(n_W):
                            vert_start = h * self.stride_shape[0]
                            vert_end = vert_start + conv_h
                            horiz_start = w * self.stride_shape[1]
                            horiz_end = horiz_start + conv_w

                            a_slice_prev = a_prev_pad[:, vert_start:vert_end, horiz_start:horiz_end]
                            # print("a_slice_prev 4d shape ", a_slice_prev.shape)

                            Z[i, c, h, w] = self.conv_single_step(a_slice_prev, self.weights[c, :, :, :],
                                                                  self.bias[np.newaxis, c, np.newaxis, np.newaxis])

                    # if n_W = 1 or x is missed in input_tensor
                    else:

                        vert_start = h * self.stride_shape[0]
                        vert_end = vert_start + conv_h
                        # horiz_start =  self.stride_shape[0]
                        # horiz_end = horiz_start
                        # print("a_prev_pad at i ", a_prev_pad.shape)
                        a_slice_prev = a_prev_pad[:, vert_start:vert_end]

                        Z[i, c, h] = self.conv_single_step(a_slice_prev, self.weights[c, :, :],
                                                           self.bias[np.newaxis, c, np.newaxis])

        return Z

    def backward(self, error_tensor):

        # Retrieve dimensions from the weights' shape
        if len(self.weights.shape) == 4:
            n_C, n_C_prev, conv_h, conv_w, = self.weights.shape  # number of kernel, c, m, n

            if conv_w % 2 == 0:
                pad_width_left = math.floor((conv_w - 1) / 2)
                pad_width_right = math.ceil((conv_w - 1) / 2)
            else:
                pad_width_left = pad_width_right = (conv_w - 1) // 2

        else:
            n_C, n_C_prev, conv_h = self.weights.shape  # number of kernel, c, m

        if len(error_tensor.shape) == 4:
            m, n_C, n_H, n_W = error_tensor.shape
        else:
            m, n_C, n_H = error_tensor.shape
            n_W = 1

        # Initialize the gradients with the same shape as the variables
        dA_prev = np.zeros_like(self.input_tensor)
        self.dW = np.zeros_like(self.weights)
        self.db = np.zeros_like(self.bias)

        if conv_h % 2 == 0:

            pad_height_top = math.floor((conv_h - 1) / 2)
            pad_height_bottom = math.ceil((conv_h - 1) / 2)

        else:
            pad_height_top = pad_height_bottom = (conv_h - 1) // 2

        if len(error_tensor.shape) == 3:
            pad_width = ((0, 0), (0, 0), (pad_height_top, pad_height_bottom))
        else:  # Assuming the input_tensor.shape is (m, n_H_prev, n_W_prev, n_C_prev) for 4D tensors
            self.pad_width = ((0, 0), (0, 0), (pad_height_top, pad_height_bottom), (pad_width_left, pad_width_right))

        A_prev_pad = np.pad(self.input_tensor, self.pad_width, mode='constant', constant_values=0)
        dA_prev_pad = np.pad(dA_prev, self.pad_width, mode='constant', constant_values=0)
        # Loop over the training examples
        for i in range(m):
            # Select the i-th example's padded error_tensor and input_tensor
            a_prev_pad = A_prev_pad[i]
            da_prev_pad = dA_prev_pad[i]
            # Loop over the  (n_C)
            for c in range(n_C):
                # Loop over the height (n_H)
                for h in range(n_H):
                    # Loop over the width (n_W)
                    if n_W > 1:
                        for w in range(n_W):
                            # Find the corners of the current "slice"
                            vert_start = h * self.stride_shape[0]
                            vert_end = vert_start + conv_h

                            horiz_start = w * self.stride_shape[1]
                            horiz_end = horiz_start + conv_w
                            # Slice the padded error_tensor and input_tensor
                            a_slice = a_prev_pad[:, vert_start:vert_end, horiz_start:horiz_end]
                            # Update gradients for the slice
                            self.dW[c, :, :, :] += a_slice * error_tensor[i, c, h, w]
                            da_prev_pad[:, vert_start:vert_end, horiz_start:horiz_end] += self.weights[c, :, :, :] * error_tensor[i, c, h, w]
                            self.db[c] += error_tensor[i, c, h, w]

                    else:

                        vert_start = h * self.stride_shape[0]
                        vert_end = vert_start + conv_h
                        a_slice = a_prev_pad[:, vert_start:vert_end]

                        self.dW[c, :, :] += a_slice * error_tensor[i, c, h]

                        da_prev_pad[:, vert_start:vert_end] += self.weights[c, :, :] * error_tensor[i, c, h]

                        self.db[c] += error_tensor[i, c, h]

            if len(error_tensor.shape) == 4:
                if self.convolution_shape[1] != 1 and self.convolution_shape[2] != 1:
                    dA_prev[i, :, :, :] = da_prev_pad[:, pad_height_top:-pad_height_bottom,
                                          pad_width_left:-pad_width_right]
                else:
                    dA_prev[i, :, :, :] = da_prev_pad

            else:
                dA_prev[i, :, :] = da_prev_pad[:, pad_height_top:-pad_height_bottom]

        if self._optimizer:
            self.weights = self.optimizer.calculate_update(self.weights, self.dW)
            self.bias = self.optimizer.calculate_update_bias(self.bias, self.db)

        return dA_prev

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer


    @property
    def gradient_weights(self):
        return self.dW

    @property
    def gradient_bias(self):
        return self.db