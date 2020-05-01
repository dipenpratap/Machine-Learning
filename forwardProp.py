from utilities import *
import numpy as np

# Initialize the network here
""" Please start by changing the parameters of the Neural Network below """
# lets start by initializing the variables in the NN as shown below
numInputs = 6
numHiddenLayers = 5
numNodesHidden = [4, 5, 5, 5, 5]
numNodesOutput = 1

my_network = initialize_network(num_inputs=numInputs, num_hidden_layers=numHiddenLayers,
                                num_nodes_hidden=numNodesHidden, num_nodes_output=numNodesOutput)


def forward_propagate(inputs):
    # global layer_outputs
    layer_inputs = list(inputs)  # start with the input layer as the input to the first hidden layer

    for layer in my_network:

        layer_data = my_network[layer]

        layer_outputs = []
        for layer_node in layer_data:
            node_data = layer_data[layer_node]

            # compute the weighted sum and the output of each node at the same time
            node_output = node_activation(compute_weighted_sum(layer_inputs, node_data['weights'], node_data['bias']))
            layer_outputs.append(np.around(node_output[0], decimals=4))

        if layer != 'output':
            print('The outputs of the nodes in hidden layer number {} is {}'.format(layer.split('_')[1], layer_outputs))

        layer_inputs = layer_outputs  # set the output of this layer to be the input to next layer

    network_predictions = layer_outputs
    return network_predictions


print(my_network)
