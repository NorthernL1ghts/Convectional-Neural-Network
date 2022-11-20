# Base class:
class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    # Compute the Y output of a layer for a given input X:
    def forward_propagation(self, input):
        raise NotImplementedError # Error handling.

    # Computes dE / dX for a given dE / dY (and update parameters if any exists):
    def backward_propagation(self, output_error, learning_rate):
        raise NotImplementedError # Error handling.