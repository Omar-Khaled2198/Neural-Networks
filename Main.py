import numpy as np
from matplotlib import pyplot as plt
from NeuralNetwork import NeuralNetwork
from Layer import Layer
np.random.seed(100)

with open('train.txt') as file:
    numOfInputs,numOfHiddenLayers,numOfOutputs = [int(x) for x in file.readline().split()]
    numOfExamples=int(file.readline())
    temp = []
    for line in file:
        temp.append(list(map(float, line.split())))
    data=np.array(temp)
    X=data[:,0:numOfInputs]
    X= X/np.max(X,axis=0)
    y=data[:,numOfInputs:numOfInputs+numOfOutputs]
    y= y/np.max(y,axis=0)

nn = NeuralNetwork()
nn.addLayer(Layer(numOfInputs,numOfInputs+1))

for i in range(numOfHiddenLayers-1):
    nn.addLayer(Layer(numOfInputs+1,numOfInputs+1))

nn.addLayer(Layer(numOfInputs+1,numOfOutputs))

# Train the neural network
errors = nn.train(X, y, 0.5, 1000)

# Plot changes in mse
plt.plot(errors)
plt.title('Changes in MSE')
plt.xlabel('Epoch (every 10th)')
plt.ylabel('MSE')
plt.show()
