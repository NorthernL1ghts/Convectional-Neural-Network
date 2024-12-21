# Base network class:
class Network:
    def __init__(self):
        self.layers = [] # Create a new empty array of layers.
        self.loss = None
        self.loss_prime = None

    # Add layer to network:
    def add(self, layer):
        self.layers.append(layer) # Add the layer.

    # Set loss to use:
    def use(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime

    # Predict output for given input:
    def predict(self, input_data):
        # Sample the dimension first
        samples = len(input_data)
        result = [] # Create an new empty array of results.

        # Run the network over all samples:
        for i in range(samples):
            # Foward propagation
            output = input_data[i] # Use "i" index for input_data.
            for layer in self.layers:
                output = layer.forward_propagation(output) # Use foward propagation for output.
            result.append(output) # Append / add the output to the result.

        return result # Return the result.

    # Train the network:
    def fit(self, x_train, y_train, epochs, learning_rate):
        # Sample the dimension first
        samples = len(x_train) # Use sample to calculate the length of x_train value.

        # Training loop:
        for i in range(epochs):
            err = 0
            for j in range(samples):
                # Foward propagation:
                output = x_train[j] # Use "j" as index for output.
                for layer in self.layers:
                    output = layer.forward_propagation(output) # Add output to foward propagation.

                # Computes loss (for display purpose only):
                err += self.loss(y_train[j], output)

                # Backward propagation:
                error = self.loss_prime(y_train[j], output)
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate) # Add's the learning_rate to backward propagation.

            # Calculate average error on all samples:
            err /= samples 
            print('epoch %d / %d   error = %f' % (i + 1, epochs, err)) # Display all the relevant information.