'''
feed forward neural network
'''
import relu_normalisation as relu
import ffnn.NeuralNet as net
import pooling_layer as flat
import numpy as np
import random,math

def node_dense(inputs,weights):
    return np.dot(inputs,weights)
def layer_dense(inputs,weights,nodes,biases):
    output = []
    for x in range(nodes):
        out = relu.relu(node_dense(inputs,weights[x])+biases[x])
        output.append(out)
    return output

def NeuralNetwork(layers,weights,biases,inputs):
    out = inputs
    for x in range(len(layers)-1):
        out = layer_dense(out,weights[x],layers[x+1],biases[x])
    return out
def tanh(x):
    return math.tanh(x)
def sigmoid(x):
    return 1 / (1 + math. exp(-x))
def init_brain(layers):
    biases = []
    weights = []
    for (index,val) in enumerate(layers[1:]):
        biases.append([random.uniform(-1,1) for x in range(val)])
        
    for (index,val) in enumerate(layers[:-1]):
        weights.append([[random.uniform(-1,1) for x in range(val)] for z in range(layers[index+1])])
    return weights,biases

