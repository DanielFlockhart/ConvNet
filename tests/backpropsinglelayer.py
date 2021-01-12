import time

def password(inp,w1,w2):
    out = relu(relu(inp * w1)*w2)
    return out
def backprop(w1,w2,loss,inp):
    global output,actual
    lossGrad = 2*(output-actual)*0.05
    actLoss = derivative(inp*w1)
    w1Loss = lossGrad/actLoss
    act2loss = derivative(lossGrad/actLoss)
    w2Loss = w1Loss/(act2loss+0.1)
    w1+=w1Loss
    w2+=w2Loss
    return w1,w2
def derivative(x):
    return 1 if x > 0 else 0

def relu(x):
    return max(x,0)

inp = 1
weight1 = 0.5
weight2 = 0.5

actual = 0.3
while True:
    output = password(inp,weight1,weight2)
    loss = (output-actual)**2
    (weight1,weight2) = backprop(weight1,weight2,loss,inp)
    time.sleep(1)
    print(loss)
