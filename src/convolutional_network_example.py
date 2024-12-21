# This file is a example of how convolutional neural networks work.
import numpy as np

from network import Network
from convolutional_layer import ConvLayer
from activation_layer import ActivationLayer
from activations import tanh, tanh_prime
from losses import mse, mse_prime

# Training data:
x_train = [np.random.rand(10,10,1)] # Create a 3D array for training x value.
y_train = [np.random.rand(4,4,2)] # Create a 3D aray for training y value.

# Network:
net = Network() # Create a new instance of the Network class.
net.add(ConvLayer((10,10,1), (3,3), 1)) # Add a new convolutional layer to the network.
net.add(ActivationLayer(tanh, tanh_prime)) # Add a new activation layer to the network
net.add(ConvLayer((8,8,1), (3,3), 1))  # Add a new convolutional layer to the network.
net.add(ActivationLayer(tanh, tanh_prime)) # Add a new activation layer to the network
net.add(ConvLayer((6,6,1), (3,3), 2))  # Add a new convolutional layer to the network.
net.add(ActivationLayer(tanh, tanh_prime)) # Add a new activation layer to the network

# Train the program:
net.use(mse, mse_prime)
net.fit(x_train, y_train, epochs = 1000, learning_rate = 0.3) # Base training parameters.

# Test the convolutional network:
out = net.predict(x_train) # Output.
print("predicted = ", out) # Print the predicted output to console (display only).
print("expected = ", y_train) # Print the expected output to console (display only).
