# This is a test file to test the created functions
# Testing the initializeNetwork function
from initializeNetwork import initializeNetwork
import numpy as np

inputs = 4
hiddenLayersNum = 3
hiddenNodesNum = [3, 4, 5]
outputNodesNum = 1
output = initializeNetwork(numInputs=inputs, numHiddenLayers=hiddenLayersNum, numNodesHidden=hiddenNodesNum,
                           numNodesOutput=outputNodesNum)
print('The output of network initializer is {}'.format(output))

# Testing the computeWeightedSum function


