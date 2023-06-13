

import numpy as np
from scipy.signal import correlate, convolve

class Conv:
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        self.trainable = True
        self.stride_shape = stride_shape if isinstance(stride_shape, tuple) else (stride_shape, )
        self.convolution_shape = convolution_shape
        self.num_kernels = num_kernels
        self.weights = np.random.uniform(size=(num_kernels, *convolution_shape))
        self.bias = np.random.uniform(size=(num_kernels,))
        self.optimizer = None

        
        

    def initialize(self, weights_initializer, bias_initializer):
        
        fan_in = np.prod(self.convolution_shape)
        fan_out = np.prod(self.convolution_shape[1:]) * self.num_kernels
        self.weights = weights_initializer.initialize(self.weights.shape, fan_in, fan_out)
        self.bias = bias_initializer.initialize(self.bias.shape, fan_in, fan_out)

    
    def forward(self, input_tensor):
        
        stride = np.array(self.stride_shape)
        if stride.ndim != 1:
            stride = np.squeeze(stride)

        if (input_tensor.shape == 3){
            batch_size, num_channels, width  = input_tensor.shape
            padding = 0
            kernel_width = self.convolution_shape[1]
            output_width = (input_width + 2 * padding - kernel_width) // stride + 1
            output_tensor = np.zeros((batch_size, self.num_kernels, output_width))
        }
        batch_size, num_channels, *spatial_dims = input_tensor.shape
        #kernel_spatial_dims = np.array(self.convolution_shape)
        
        # to avoid this error: only integer scalar arrays can be converted to a scalar index when stride_shape =[2]
        
        print("stride ",stride)

        output_spatial_dims = (spatial_dims + stride - 1) // stride
        
        print("spatial_dims * ", *spatial_dims)
        print("spatial_dims ", spatial_dims)
        print("output_spatial_dims ",*output_spatial_dims, "length dim ", len(spatial_dims))
        #print("output_spatial_dims * ",*output_spatial_dims.shape)
        output_tensor = np.zeros((batch_size, self.num_kernels, *output_spatial_dims))
        print("output_tensor ", output_tensor.shape)
        for b in range(batch_size):
            for k in range(self.num_kernels):
                for c in range(num_channels):
                    if len(spatial_dims) == 1:  # If input tensor is 1D
                        input_reshaped = input_tensor[b, c]
                        weights_reshaped = self.weights[k, c].reshape(-1)
                        
                        
                        output_tensor[b, k] += correlate(input_tensor[b, c], self.weights[k, c], mode='same')
                    else:  # If input tensor is 2D
                        output_tensor[b, k] += correlate(input_tensor[b, c], self.weights[k, c], mode='same')
                output_tensor[b, k] += self.bias[k]
                
                output_tensor[b, k] = output_tensor[b, k, ::self.stride_shape[0], ::self.stride_shape[1]]
        print("----------------")
        return output_tensor

    

    def backward(self, error_tensor):
        batch_size, num_channels, *spatial_dims = error_tensor.shape
        spatial_dims = np.array(spatial_dims)
        kernel_spatial_dims = np.array(self.convolution_shape[1:])
        stride = np.array(self.stride_shape)
        input_spatial_dims = (spatial_dims - 1) * stride + kernel_spatial_dims
        input_error_tensor = np.zeros((batch_size, *self.convolution_shape))
        self._gradient_weights = np.zeros_like(self.weights)
        self._gradient_bias = np.zeros_like(self.bias)

        for b in range(batch_size):
            for k in range(self.num_kernels):
                upsampled_error_tensor = np.zeros((input_spatial_dims[0], input_spatial_dims[1]))
                upsampled_error_tensor[::stride[0], ::stride[1]] = error_tensor[b, k]
                for c in range(self.convolution_shape[0]):
                    input_error_tensor[b, c] += convolve(upsampled_error_tensor, self.weights[k,c], mode='same')
                    self._gradient_weights[k,c] += correlate(upsampled_error_tensor, error_tensor[b,k], mode='valid')
                self._gradient_bias[k] += error_tensor[b,k].sum()

        if self.optimizer:
            self.weights -= self.optimizer.optimize(self.gradient_weights)
            self.bias -= self.optimizer.optimize(self.gradient_bias)

        return input_error_tensor

   

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer


import numpy as np
from scipy.signal import correlate, convolve

class Conv:
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        self.trainable = True
        self.stride_shape = stride_shape if isinstance(stride_shape, tuple) else (stride_shape, stride_shape)
        self.convolution_shape = convolution_shape
        self.num_kernels = num_kernels
        self.weights = np.random.uniform(size=(num_kernels, *convolution_shape))
        self.bias = np.random.uniform(size=(num_kernels,))
        self.optimizer = None

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @property
    def gradient_bias(self):
        return self._gradient_bias

    def forward(self, input_tensor):
        
        batch_size, num_channels, *spatial_dims = input_tensor.shape
        
        spatial_dims = np.array(spatial_dims)
        
        kernel_spatial_dims = np.array(self.convolution_shape[1:])
        
        stride = np.array(self.stride_shape)
        # check whehther input self.stride_shape dimension is not 1D
        if stride.ndim != 1:
            stride = np.squeeze(stride)
        #print("stride ", stride, "its shape ", stride.shape)
        output_spatial_dims = (spatial_dims + stride - 1) // stride
        
        
        
        output_tensor = np.zeros((batch_size, self.num_kernels, *output_spatial_dims))
        print("self.weights shape ", self.weights.shape)
        for b in range(batch_size):
            for k in range(self.num_kernels):
                for c in range(num_channels):
                    print("input_tensor[b,c] shape ", input_tensor[b, c].shape)
                    print ("self.weights[k, c] ", self.weights[k, c].shape)
                    print("output_tensor[b, k] ", output_tensor[b, k].shape)

                    if len(spatial_dims) == 1: #if input tensor is 1D
                        input_reshaped = input_tensor[b, c].reshape((*spatial_dims, 1))
                        output_tensor[b, k] += np.correlate(input_reshaped, self.weights[k, c], mode='same')
                    else:
                        output_tensor[b, k] += correlate(input_tensor[b, c], self.weights[k, c], mode='same')
                    
                output_tensor[b, k] += self.bias[k]
                output_tensor[b, k] = output_tensor[b, k, ::self.stride_shape[0], ::self.stride_shape[1]]
        print("-----------------")
        return output_tensor

import numpy as np
from scipy.signal import correlate, convolve

class Conv:
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        self.trainable = True
        self.stride_shape = stride_shape if isinstance(stride_shape, tuple) else (stride_shape, stride_shape)
        self.convolution_shape = convolution_shape
        self.num_kernels = num_kernels
        self.weights = np.random.uniform(size=(num_kernels, *convolution_shape))
        self.bias = np.random.uniform(size=(num_kernels,))
        self.optimizer = None

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @property
    def gradient_bias(self):
        return self._gradient_bias

    def forward(self, input_tensor):
        batch_size, num_channels, *spatial_dims = input_tensor.shape
        spatial_dims = np.array(spatial_dims)
        kernel_spatial_dims = np.array(self.convolution_shape[1:])
        stride = np.array(self.stride_shape)
        output_spatial_dims = (spatial_dims + stride - 1) // stride
        output_tensor = np.zeros((batch_size, self.num_kernels, *output_spatial_dims))

        for b in range(batch_size):
            for k in range(self.num_kernels):
                for c in range(num_channels):
                    output_tensor[b, k] += correlate(input_tensor[b, c], self.weights[k, c], mode='same')
                output_tensor[b, k] += self.bias[k]
                output_tensor[b, k] = output_tensor[b, k, ::self.stride_shape[0], ::self.stride_shape[1]]

        return output_tensor

    def forward(self, input_tensor):
        print("input_tensor shape ", input_tensor.shape)
        batch_size, num_channels, *spatial_dims = input_tensor.shape
        print("input_tensor.shape ", input_tensor.shape)
        print("*spatial_dims ", *spatial_dims)
        spatial_dims = np.array(spatial_dims)
        
        kernel_spatial_dims = np.array(self.convolution_shape[1:])
        stride = np.array(self.stride_shape)
        output_spatial_dims = (spatial_dims + stride - 1) // stride
        print("output_spatial_dims ", *output_spatial_dims)
        output_tensor = np.zeros((batch_size, self.num_kernels, *output_spatial_dims))

        for b in range(batch_size):
            for k in range(self.num_kernels):
                for c in range(num_channels):
                    output_tensor[b, k] += correlate(input_tensor[b, c], self.weights[k, c], mode='same')
                output_tensor[b, k] += self.bias[k]
                output_tensor[b, k] = output_tensor[b, k, ::self.stride_shape[0], ::self.stride_shape[1]]

        return output_tensor

        return input_error_tensor

    def initialize(self, weights_initializer, bias_initializer):
        self.weights = weights_initializer.initialize(self.weights.shape)
        self.bias = bias_initializer.initialize(self.bias.shape)





import numpy as np
from scipy.signal import correlate, correlate2d

class Conv:
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        self.stride_shape = stride_shape
        self.convolution_shape = convolution_shape
        self.num_kernels = num_kernels
        self.trainable = True
        self.weights = np.random.uniform(size=(num_kernels, *convolution_shape))
        self.bias = np.random.uniform(size=(num_kernels,))
        self._gradient_weights = None
        self._gradient_bias = None
        self._optimizer = None

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @property
    def gradient_bias(self):
        return self._gradient_bias

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer

    def forward(self, input_tensor):
        batch_size, input_channels = input_tensor.shape[:2]
        input_shape = input_tensor.shape[2:]

        if len(input_shape) == 1:
            conv_func = correlate
        elif len(input_shape) == 2:
            conv_func = correlate2d
        else:
            raise ValueError("Input tensor dimension must be 1D or 2D.")

        output_shape = self.calculate_output_shape(input_shape)
        output_tensor = np.zeros((batch_size, self.num_kernels, *output_shape))

        for i in range(batch_size):
            for j in range(self.num_kernels):
                for k in range(input_channels):
                    kernel = self.weights[j, k]
                    stride = self.stride_shape if isinstance(self.stride_shape, int) else self.stride_shape[0]

                    if len(input_shape) == 1:
                        padded_input = np.pad(input_tensor[i, k], (kernel.shape[0] - 1, 0), mode='constant')
                        output_tensor[i, j] += conv_func(padded_input, kernel, mode='valid', method='direct')[::stride]
                    elif len(input_shape) == 2:
                        padded_input = np.pad(input_tensor[i, k], ((kernel.shape[0] - 1, 0), (kernel.shape[1] - 1, 0)), mode='constant')
                        output_tensor[i, j] += conv_func(padded_input, kernel, mode='valid', method='direct')[::stride, ::stride]

                output_tensor[i, j] += self.bias[j]

        return output_tensor

    def backward(self, error_tensor):
        batch_size, output_channels = error_tensor.shape[:2]
        output_shape = error_tensor.shape[2:]

        if len(output_shape) == 1:
            conv_func = correlate
        elif len(output_shape) == 2:
            conv_func = correlate2d
        else:
            raise ValueError("Output tensor dimension must be 1D or 2D.")

        input_shape = self.calculate_input_shape(output_shape)
        input_tensor = np.zeros((batch_size, self.weights.shape[1], *input_shape))

        self._gradient_weights = np.zeros_like(self.weights)
        self._gradient_bias = np.sum(error_tensor, axis=(0, 2, 3))

        for i in range(batch_size):
            for j in range(output_channels):
                for k in range(self.weights.shape[1]):
                    kernel = np.flip(self.weights[j, k], axis=(0, 1))
                    stride = self.stride_shape if isinstance(self.stride_shape, int) else self.stride_shape[0]

                    if len(output_shape) == 1:
                        padded_error = np.pad(error_tensor[i, j], (0, kernel.shape[0] - 1), mode='constant')
                        input_tensor[i, k] += conv_func(padded_error, kernel, mode='valid', method='direct')[::stride]
                    elif len(output_shape) == 2:
                        padded_error = np.pad(error_tensor[i, j], ((0, kernel.shape[0] - 1), (0, kernel.shape[1] - 1)), mode='constant')
                        input_tensor[i, k] += conv_func(padded_error, kernel, mode='valid', method='direct')[::stride, ::stride]

                    self._gradient_weights[j, k] += conv_func(padded_error, input_tensor[i, k], mode='valid', method='direct')

        if self.optimizer:
            self.weights -= self.optimizer.learning_rate * self._gradient_weights
            self.bias -= self.optimizer.learning_rate * self._gradient_bias

        return input_tensor

    def calculate_output_shape(self, input_shape):
        if isinstance(self.stride_shape, int):
            stride_shape = (self.stride_shape,) * len(input_shape)
        else:
            stride_shape = self.stride_shape

        output_shape = []
        for i, dim in enumerate(input_shape):
            output_dim = (dim - self.convolution_shape[i] + stride_shape[i]) // stride_shape[i]
            output_shape.append(output_dim)

        return tuple(output_shape)

    def calculate_input_shape(self, output_shape):
        if isinstance(self.stride_shape, int):
            stride_shape = (self.stride_shape,) * len(output_shape)
        else:
            stride_shape = self.stride_shape

        input_shape = []
        for i, dim in enumerate(output_shape):
            input_dim = (dim - 1) * stride_shape[i] + self.convolution_shape[i]
            input_shape.append(input_dim)

        return tuple(input_shape)
    



    def forward(self, input_tensor):
        
        stride = np.array(self.stride_shape)
        if stride.ndim != 1:
            stride = np.squeeze(stride)

        
        batch_size, num_channels, *spatial_dims = input_tensor.shape
        #kernel_spatial_dims = np.array(self.convolution_shape)
        
        # to avoid this error: only integer scalar arrays can be converted to a scalar index when stride_shape =[2]
        
        print("stride ",stride)

        output_spatial_dims = (spatial_dims + stride - 1) // stride
        
        print("spatial_dims * ", *spatial_dims)
        #print("spatial_dims ", spatial_dims)
        #print("output_spatial_dims ",*output_spatial_dims, "length dim ", len(spatial_dims))
        #print("output_spatial_dims * ",*output_spatial_dims.shape)
        output_tensor = np.zeros((batch_size, self.num_kernels, *output_spatial_dims))
        print("output_tensor ", output_tensor.shape)
        for b in range(batch_size):
            for k in range(self.num_kernels):
                for c in range(num_channels):
                    if len(spatial_dims) == 1:  # If input tensor is 1D
                        #input_reshaped = input_tensor[b, c]
                        weights_reshaped = self.weights[k, c].reshape(-1)
                        
                        
                        output_tensor[b, k] += correlate(input_tensor[b, c], self.weights[k, c], mode='same')
                    else:  # If input tensor is 2D
                        output_tensor[b, k] += correlate(input_tensor[b, c], self.weights[k, c], mode='same')
                
                if len(spatial_dims) == 1:  # If input tensor is 1D
                    output_tensor[b, k] = output_tensor[b, k, ::self.stride_shape[0]]
                else:
                    #print("'stride len ", len)
                    output_tensor[b, k] = output_tensor[b, k, ::self.stride_shape[0], ::self.stride_shape[1]]
                output_tensor[b, k] += self.bias[k]
        print("----------------")
        return output_tensor





        from Layers.Base import BaseLayer
import numpy as np




class Pooling(BaseLayer):
    def __init__(self, stride_shape, pooling_shape):
        
        super().__init__()
        self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape
        self.input_tensor_shape = None
        self.trainable = False

    def forward(self, input_tensor):

        self.input_tensor = input_tensor
        self.input_tensor_shape = input_tensor.shape
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
        error_tensor_prev = np.zeros(self.input_tensor_shape)

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
                        error_tensor_prev[b, c, vert_start:vert_end, horiz_start:horiz_end] = mask * error_tensor[b, c, h, w]
        
        return error_tensor_prev