import numpy as np
from PIL import Image, ImageOps
import random,time
import relu_normalisation as relu
import convolution_layer as c_layer
import pooling_layer as p_layer

def get_data(name,size):
    img = Image.open('data/'+name+ '.png').convert('L')
    img.thumbnail(size)
    img_inverted = ImageOps.invert(img)
    np_img = np.array(img_inverted)
    np_img[np_img > 0] = 1
    return np_img


def initialise_image(name,catergory,window_size,stride,size):
    img = get_data(f"{catergory}//{name}",size)
    if(catergory == "circles"):
        actual = [0,1,0]
    if(catergory == "triangles"):
        actual = [0,0,1]
    if(catergory == "squares"):
        actual = [1,0,0]
    convolutions = c_layer.process(size,img)
    pooled_convolutions = p_layer.pool(window_size,stride,convolutions)
    normalised_convolutions = relu.normalise(pooled_convolutions)
    inputs = p_layer.inputify(normalised_convolutions)
    return (inputs,actual)
