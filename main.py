'''
convolution layer
RelU Layer - Normalisation
Pooling layer
feed forward layer

Backpropogation and loss calculation
'''


import relu_normalisation as relu
import backprop_layer as b_layer
import convolution_layer as c_layer
import feed_forward_layer as ff_layer
import convert_image as init_data
import train
import model_data as model
import numpy as np
import random,time


img_size = (8,8)
stride = 2
window_size = 2
pools = 2
layers = [192,128,128,128,3]
shapes = ["square","circle","triangle"]
def ran(maxn,minn):
    return random.uniform(maxn,minn)

def create_test_case(size):
    (width,height) = size
    array = np.array([[ran(-1,1) for x in range(width)] for z in range(height)])
    return array
    
def run(stride,window_size,name,model_name,img_size,layers,shapes,catergory):
    (weights,biases) = model.load_model(model_name)
    (image,expected) = init_data.initialise_image(name,catergory,window_size,stride,img_size)
    return train.one_shot(weights,biases,image,expected,layers,shapes)

if __name__ == "__main__":
    epochs = int(input("Epochs: "))
    train.train(epochs,layers,stride,window_size,img_size,shapes,20)
    
    #print(f"Training Complete")
##    model_name = str(input("Model name: "))
##    while True:
##        image_name = str(input("Image name: "))
##        catergory = str(input("Catergory name: "))
##        result = run(stride,window_size,image_name,model_name,img_size,layers,shapes,catergory)
##        print(f'Outputs : {result[0]}')
##        print(f'Prediction : {result[1][0]}')
##        print(f'Loss: {result[2]}%')
        
    
    
