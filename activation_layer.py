# An activation function in a neural network defines how the weighted sum of the input is transformed into an output,
# from a node or nodes in a layer of the network.
from layer import Layer

# Inherit from base class Layer:
class ActivationLayer(Layer):
    def __init__(self, activation, activation_prime):
        self.activation = activation # Activation.
        self.activation_prime = activation_prime # Prime activation.

    # Returns the activated input:
    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output # Returns the output.

    # Returns input_error = dE / dX for a given output_error = dE / dY,
    # learning_rate is not used because there is no "learnable" parameters.
    def backward_propagation(self, output_error, learning_rate):
        return self.activation_prime(self.input) * output_error # Return the output using prime activation.