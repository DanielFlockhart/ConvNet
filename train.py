'''

'''
import feed_forward_layer as ff_layer
import backprop_layer as b_layer
import convert_image as image
import model_data as model
import batch_data
import time


def get_prediction(weights,possibles):
    max_index = weights.index(max(weights))
    return (possibles[max_index], max(weights))


def one_shot(weights,biases,inputs,expected,layers,shapes):
    feed_forward = ff_layer.NeuralNetwork(layers,weights,biases,inputs)
    prediction = get_prediction(feed_forward,shapes)
    loss = b_layer.loss(expected,feed_forward)
    return (feed_forward,prediction,loss)


def get_img_name(catergory,num):
    return catergory+str(num)

def get_catergory(shape):
    catergorys = ["circles","triangles","squares"]
    cat = ""
    if shape[0] == "c":
        cat = catergorys[0]
    elif shape[0] == "t":
        cat = catergorys[1]
    elif shape[0] == "s":
        cat = catergorys[2]
    else:
        raise "Get Catergory not understood"
    return cat

def batch_pass(shapes,weights,biases,layers,stride,window_size,img_size,batch):
    loss_total = [0,0,0]
    for (index, shape) in enumerate(batch):
        (inputs,expected) = image.initialise_image(shape,get_catergory(shape),window_size,stride,img_size)
        (model,prediction,loss) = one_shot(weights,biases,inputs,expected,layers,shapes)
        loss_total = [loss[x] + loss_total[x] for x in range(len(loss_total))]
    loss_total = b_layer.average_change(loss_total,len(batch))
    return loss_total



def train(epochs,layers,stride,window_size,size,shapes,batch_size):
    (best_model,best_loss) = ([],10000)
    (weights,biases) = ff_layer.init_brain(layers
    for x in range(epochs):
        batch = batch_data.get_batch(batch_size,100,shapes)
        loss = batch_pass(shapes,weights,biases,layers,stride,window_size,size,batch)
        (weights,biases) = b_layer.backwards_pass(weights,biases,loss,layers)
        print((loss[0]+loss[1]+loss[2])/3)
        print(len(weights[0]))
    model_name = str(input("Model Name : "))
    model.save_model("model0",(weights,biases))
