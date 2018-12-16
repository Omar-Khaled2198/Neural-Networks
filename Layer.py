import numpy as np
class Layer:

    def __init__(self, n_input, n_neurons, weights=None, bias=None):

        # int n_input: The input size (coming from the input layer or a previous hidden layer)
        # int n_neurons: The number of neurons in this layer.
        # weights: The layer's weights.
        # bias: The layer's bias.
        self.weights = weights if weights is not None else np.random.rand(n_input, n_neurons)
        self.bias = bias if bias is not None else np.random.rand(n_neurons)
        self.last_activation = None
        self.error = None
        self.delta = None



    def activate(self, x):

        # Calculates the dot product of this layer.
        r = np.dot(x, self.weights) + self.bias
        self.last_activation = 1 / (1 + np.exp(-r))
        return self.last_activation


    def activationDerivative(self, r):

        #the derivative of the activation function
        return r * (1 - r)


