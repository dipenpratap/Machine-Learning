# This is a test file to test the created functions
# Testing the initializeNetwork function
from initializeNetwork import initializeNetwork
import numpy as np
from computeWeightedSum import computeWeightedSum
from activationCalc import activationCalc

inputs = 5
hiddenLayersNum = 3
hiddenNodesNum = [3, 4, 5]
outputNodesNum = 1
smallNetwork = initializeNetwork(numInputs=inputs, numHiddenLayers=hiddenLayersNum, numNodesHidden=hiddenNodesNum,
                                 numNodesOutput=outputNodesNum)
print('The output of network initializer is {}'.format(smallNetwork))

# Testing the computeWeightedSum function

# lets generate 5 inputs that can be fed into the network
testInputs = np.around(np.random.uniform(size=5), decimals=2)
print('The 5 test inputs are {}'.format(testInputs))

# Now lets compute the weighted sum at the first node of first layer of the network
node1Weights = smallNetwork['Layer1']['node1']['weights']
node1Bias = smallNetwork['Layer1']['node1']['bias']
node1WeightedSum = computeWeightedSum(testInputs, node1Weights, node1Bias)
print('The weighted Sum of node {} in layer {} is {}'.format('1', '1', node1WeightedSum[0]))


# Now lets calculate the activation of the weighted sum of node 1 in layer 1
node1Activation = activationCalc(node1WeightedSum)
print('The activation of node 1 in layer 1 is {}'.format(node1Activation[0]))

