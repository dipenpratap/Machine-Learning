# This function computes the weighted sum of each node
import numpy as np


def computeWeightedSum(inputs, weights, bias):
    return np.sum(inputs * weights) + bias
