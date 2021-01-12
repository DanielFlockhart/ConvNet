'''
Normalisation layer
'''

import numpy as np
import random
import main


def ReLU_normalisation(array):
    for x in range(len(array)):
        array[x] = relu(array[x])
    return array
def relu(x):
   return np.maximum(0,x)
def normalise(arrayList):
    normalised = []
    for x in arrayList:
        normalised.append(ReLU_normalisation(x))
    return normalised
    
        
    
