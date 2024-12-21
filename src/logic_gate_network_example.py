# This file is a digital logic gate that gives a true output only when both its inputs differ from each other.
import numpy as np

from network import Network
from fully_connected_layer import FCLayer
from activation_layer import ActivationLayer
from activations import tanh, tanh_prime
from losses import mse, mse_prime

# Training data:
x_train = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])  # Create a 3D array for training x value.
y_train = np.array([[[0]], [[1]], [[1]], [[0]]])  # Create a 3D array for training y value.

# Network:
net = Network() # Create a new instance of the network class.
net.add(FCLayer(2, 3)) # Add a new Fully connected layer to the network.
net.add(ActivationLayer(tanh, tanh_prime)) # Add a new activation layer to the network.
net.add(FCLayer(3, 1)) # Add a new Fully connected layer to the network.
net.add(ActivationLayer(tanh, tanh_prime)) # Add a new activation layer to the network.

# Train the program:
net.use(mse, mse_prime)
net.fit(x_train, y_train, epochs = 1000, learning_rate = 0.1)

# Test the program:
out = net.predict(x_train) # Output.
print(out) # Print the output the console (display).
