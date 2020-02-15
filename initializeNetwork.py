import numpy as np


def initializeNetwork(numInputs, numHiddenLayers, numNodesHidden, numNodesOutput):
    # lets start by initializing the variables in the NN
    # numInputs = 2
    # numHiddenLayers = 2
    # numNodesHidden = [2, 2]
    # numNodesOutput = 1

    # import numpy as np

    numNodesPrevious = numInputs  # number of nodes in the previous layer
    network = {}  # Initializing the network

    """ Loop through each layer and randomly initialize the weights and biases for
        each nodes in the network 
        Also, notice that to represent the output layer we have to add 1 to the number
        of hidden layers """

    for layer in range(numHiddenLayers + 1):
        # lets name the layers
        if layer == numHiddenLayers:
            layerName = 'Output layer'
            numNodes = numNodesOutput
        else:
            layerName = 'Layer{}'.format(layer + 1)
            numNodes = numNodesHidden[layer]

            # Now lets initialize the weights and biases associated with each node in the current layer
            network[layerName] = {}
            for node in range(numNodes):
                nodeName = 'node{}'.format(node + 1)
                network[layerName][nodeName] = dict(
                    weights=np.around(np.random.uniform(size=numNodesPrevious), decimals=3),
                    bias=np.around(np.random.uniform(size=1), decimals=3))
            numNodesPrevious = numNodes

    return network
