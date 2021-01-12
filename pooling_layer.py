'''
Pooling layer
'''
import numpy as np
import itertools
def pool(size,stride,lists):
    pooled_list = []
    for x in lists:
        pooled_list.append(walk(x,size,stride))
    return pooled_list

def walk(sLis,size,stride):
    pooled = []
    xPos = 0
    yPos = 0
    while yPos < len(sLis):
        
        while xPos < len(sLis[0]):
            vals = [sLis[yPos][xPos],sLis[yPos][xPos+1],sLis[yPos+1][xPos],sLis[yPos+1][xPos+1]]
            pooled.append(get_max(vals))
            xPos +=2
        xPos = 0
        yPos += 2
    return pooled

def flattenise(array):
    return array.flatten()
def inputify(array):
    return [j for sub in array for j in sub]
def reshapeIT(listTo,size):
    return listTo.reshape(-1,size)

def get_max(array):
    return max(array)

