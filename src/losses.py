import numpy as np

# Loss function and its derivative:
def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2)) # Calculate and return the y prediction.

def mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.size # Calculates and returns the y prediction and divide it by the size of value y.
