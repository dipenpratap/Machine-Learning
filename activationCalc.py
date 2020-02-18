# Activation function of the NN
"""
In this case, we are going to use Sigmoid as our activation funtion to make
the calculation easy and intuitive...
"""
import numpy as np


def activationCalc(weightedSum):
    activation = 1 / (1 + np.exp(-1 * weightedSum))
    return activation
