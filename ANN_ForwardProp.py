
""" lets assume we have a simple ANN with two hidden layers;
input layer has two inputs and output layer has one output
There would be total of 6 connections assuming the first hidden layer has two neurons
and the second hidden layer has only one neuron.
Now we have 6 weights as there would be 6 connections.
three biases for three different neurons in the hidden layers
"""
# importing libraries (dependencies)
import numpy as np

# Assigning random uniform values to the weights of the NN
weights = np.around(np.random.uniform(size=6), decimals=2)
biases = np.around(np.random.uniform(size=3), decimals=2)
# lets print these values to make sure they are legal and for sanity check
print('The weights are {} and the biases are {}'.format(weights, biases))

# Now lets have two inputs to feed in into the NN
x1 = 5
x2 = 0.3

# Now lets calculate the weighted sum of the inputs in hidden layer 1
z11 = x1 * weights[0] + x2 * weights[1] + biases[0]
print('The weighted sum of {} is {}'.format('z11', np.around(z11, decimals=3)))

z12 = x1 * weights[2] + x2 * weights[3] + biases[1]
print('The weighted sum of {} is {}'.format('z12', np.around(z12, decimals=3)))

# Now lets assume that the activation function for this FFNN is a sigmoid function.
''' lets compute the activation of the fist and second neurons in the first
hidden layer. 
General formula for calculating the sigmoid is 
     Sigmoid(x) = 1/(1+exp(-x))
     '''
a11 = 1 / (1 + np.exp(-z11))
a12 = 1 / (1 + np.exp(-z12))
print('The activation of {} and {} are {} and {} respectively.'
      .format('z11', 'z12', np.around(a11, decimals=3), np.around(a12, decimals=3)))

# the outputs in hidden layer 1 becomes the outputs for hidden layer 2
''' Inputs for hidden layer 2 are the activation functions calculated using z11 and z12
'''
# calculating the weighted value for hidden layer 2
z2 = a11 * weights[4] + a12 * weights[5] + biases[2]
print('The weighted sum of {} is {}'.format('z2', np.around(z2, decimals=3)))

# calculating the activation value for z2
a2 = 1 / (1 + np.exp(-z2))
print('The output of this FFNN is {}'.format(np.around(a2, decimals=3)))
