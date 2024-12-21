# This file is a convolutional block for the Neural Network:
from layer import Layer
from scipy import signal
import numpy as np

# Inherit from base class Layer.
# This convolutional layer is always with stride 1:
class ConvLayer(Layer):
    # input_shape = (i,j,d)
    # kernel_shape = (m,n)
    # layer_depth = output_depth.
    def __init__(self, input_shape, kernel_shape, layer_depth):
        self.input_shape = input_shape 
        self.input_depth = input_shape[2]
        self.kernel_shape = kernel_shape
        self.layer_depth = layer_depth
        self.output_shape = (input_shape[0] - kernel_shape[0] + 1, input_shape[1] - kernel_shape[1] + 1, layer_depth) # Calculate the output's shape.
        self.weights = np.random.rand(kernel_shape[0], kernel_shape[1], self.input_depth, layer_depth) - 0.5 # Calculate the weights.
        self.bias = np.random.rand(layer_depth) - 0.5 # Calculate the bias.

    # Returns output for a given input:
    def forward_propagation(self, input):
        self.input = input
        self.output = np.zeros(self.output_shape)

        for k in range(self.layer_depth):
            for d in range(self.input_depth):
                self.output[:,:,k] += signal.correlate2d(self.input[:,:,d], self.weights[:,:,d,k], 'valid') + self.bias[k] # Calculate the correlated 2D and returns output.

        return self.output # Returns output.

    # Computes dE / dW, dE / dB for a given output_error = dE / dY. 
    # Returns input_error = dE / dX.
    def backward_propagation(self, output_error, learning_rate):
        in_error = np.zeros(self.input_shape)
        dWeights = np.zeros((self.kernel_shape[0], self.kernel_shape[1], self.input_depth, self.layer_depth)) # Calculate the distributed wieghts.
        dBias = np.zeros(self.layer_depth)

        for k in range(self.layer_depth):
            for d in range(self.input_depth):
                in_error[:,:,d] += signal.convolve2d(output_error[:,:,k], self.weights[:,:,d,k], 'full') # Convolve the 2D weights.
                dWeights[:,:,d,k] = signal.correlate2d(self.input[:,:,d], output_error[:,:,k], 'valid') # Correlate the 2d values of the distributed weights.
            dBias[k] = self.layer_depth * np.sum(output_error[:,:,k]) # The distributed weights use layer_depth and sum of output.

        self.weights -= learning_rate * dWeights # Calculate the weights.
        self.bias -= learning_rate * dBias # Calculate the bias.
        return in_error # Return error.