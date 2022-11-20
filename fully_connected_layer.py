# This file is a a fully connected layer multiplies the input by a weight matrix and then adds a bias vector. 
# The convolutional layers are followed by one or more fully connected layers. 
# As the name suggests, all neurons in a fully connected layer connect to all the neurons in the previous layer.    
from layer import Layer
import numpy as np

# inherit from base class Layer
class FCLayer(Layer):
    # input_size = number of input neurons
    # output_size = number of output neurons
    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(input_size, output_size) - 0.5 # Calcualate the weights.
        self.bias = np.random.rand(1, output_size) - 0.5 # Calculate the bias.

    # Returns the output for a given input:
    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias # Calculates the output.
        return self.output

    # Compute dE / dW, dE / dB for a given output_error = dE / dY. 
    # Returns input_error = dE / dX.
    def backward_propagation(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.weights.T) # Error handling
        weights_error = np.dot(self.input.T, output_error) # Error handling
        # dBias = output_error

        # Update the necessary parameters:
        self.weights -= learning_rate * weights_error # Calcualte the new weights.
        self.bias -= learning_rate * output_error # Calculate the new bias. 
        return input_error # Returns the error handling.