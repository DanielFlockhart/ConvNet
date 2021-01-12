'''
Backprop learning layer


Change weights
Change Biases
(Proportional to how far away loss is)
Change activations
'''
import math
import torch
def backwards_pass(weights,biases,loss,layers):
    ind = loss.index(max([abs(x) for x in loss]))
    for x in range(len(biases)):
        biases[x] = update_weights(biases[x],loss[ind])
    return (weights,biases)
    
def loss(actual,result):
    losses = [0,0,0]
    for x in range(len(actual)):
        losses[x] = (actual[x]-result[x])**2
    return losses


def average_change(losses,batch_size):
    return [losses[x] / batch_size for x in range(len(losses))]

'''
Determine how sensitive the loss/cost function is to weights and biases
Get most bang for buck in changes when updating weights and in turn changing the loss
find the derivative of the cost relative to the weight/bias

where dz is the derivative of the activation function
a is the activation 
dC    dz  da  dc
___ = ___ ___ ___
dw    dw  dz  da


'''
def update_weights(weights,loss):
    new = [x+calculate_step(loss) for x in weights]
    return new

def chain_rule(start_layer):
    val = get_gradient(dx,dy)


def get_gradient(dy,dx):
    pass
def calculate_step(loss):
    return 0.001 * loss

