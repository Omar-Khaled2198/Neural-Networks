import numpy as np

class NeuralNetwork:


    def __init__(self):
        self.layers = []

    def addLayer(self, layer):

        #Adds a layer to the neural network.
        self.layers.append(layer)

    def feedForward(self, X):

        #Feed forward the input through the layers.
        for layer in self.layers:
            X = layer.activate(X)

        return X

    def backPropagation(self, X, y):

        # apply the backward propagation algorithm.
        # X: The input values.
        # y: The target values.
        # learning_rate: The learning rate (between 0 and 1).
        # Loop over the layers backward
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            if layer == self.layers[-1]:
                layer.error = y - X
                layer.delta = layer.error * layer.activationDerivative(X)
            else:
                next_layer = self.layers[i + 1]
                layer.error = np.dot(next_layer.weights, next_layer.delta)
                layer.delta = layer.error * layer.activationDerivative(layer.last_activation)

    # Update the weights
    def updateWeights(self,X,learning_rate):
        for i in range(len(self.layers)):
            layer = self.layers[i]
            # The input is either the previous layers output or X itself (for the first hidden layer)
            input_to_use = np.atleast_2d(X if i == 0 else self.layers[i - 1].last_activation)
            layer.weights += layer.delta * input_to_use.T * learning_rate

    def train(self, X, y, learning_rate, max_epochs):

        # Trains the neural network using backPropagation.
        # X: The input values.
        # y: The target values.
        # learning_rate: The learning rate.
        # max_epochs: The maximum number of epochs (iterations).
        mses = []
        for i in range(max_epochs):
            for j in range(len(X)):
                output = self.feedForward(X[j])
                self.backPropagation(output, y[j])
                self.updateWeights(X[j],learning_rate)
            if i % 10 == 0:
                mse = np.mean(np.square(y - self.feedForward(X)))
                mses.append(mse)
                print('Epoch: #%s, MSE: %f' % (i, float(mse)))

        return mses
