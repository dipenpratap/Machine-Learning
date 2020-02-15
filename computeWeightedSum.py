# This function computes the weighted sum of each node

def computeWeightedSum(inputs, weights, bias):
    return np.sum(inputs * weights) + bias
