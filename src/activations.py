import numpy as np

# Activation function and its derivative:
def tanh(x):
    return np.tanh(x) # Returns the tangent height of value x.

def tanh_prime(x):
    return 1 - np.tanh(x) ** 2 # Returns the tangent height of value x^2.