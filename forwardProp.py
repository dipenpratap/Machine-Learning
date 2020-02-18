def forwardProp(network,inputs):
    layerInputs = list(inputs)
    for layer in network:
        layerData = network[layer]
        layerOutputs = []

        for layerNode in layerData:
            nodeData = layerData[layerNode]

            # Now lets compute the weighted sum at each node in the NN
