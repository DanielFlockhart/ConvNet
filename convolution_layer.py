'''
Convolution layer

Filters:
- Straight line; (up, down,left,right)*3 for each orientation
- Diagonal line; l-r,r-l

'''
import numpy as np

def dot_product_filter(array,filt,size,count):
    val = sum(sum([array[x] * filt[x] for x in range(size)]))
    return val/count
def rotate_filter(filt,times):
    for x in range(times):
        filt = np.rot90(filt)
    return filt

def get_convolutions(filt,img):
    values= dot_product_filter(img,filt,3,9)
    return values

def get_portion(image,start):
    (x,y) = start
    posList =[[[x-1,y-1],[x,y-1],[x+1,y-1]],
        [[x-1,y  ],[x,y  ],[x+1,y  ]],
        [[x-1,y+1],[x,y+1],[x+1,y+1]]]
    
    newList = [[[],[],[]],[[],[],[]],[[],[],[]]]
    for y in range(len(posList)):
        for x in range(len(posList[y])):
            yCond = posList[y][x][1]
            xCond = posList[y][x][0]
            try:
                newList[y][x] = image[yCond][xCond]
            except:
                newList[y][x] = []
                break
            if min(posList[y][x]) < 0 or y > len(posList)-1 or x > len(posList[x])-1:
                newList[y][x] = 0
            else:
                newList[y][x] = image[yCond][xCond]
    for y in range(len(newList)):
        for x in range(len(newList[y])):
            if newList[y][x] == list([]):
                newList[y][x] = 0
            
    return newList



# Straight Filters
straight1 = np.array([[1,-1,-1],
                      [1,-1,-1],
                      [1,-1,-1]])
straight2 = np.array([[-1,1,-1],
                      [-1,1,-1],
                      [-1,1,-1]])
straight3 = rotate_filter(straight1,1)
straight4 = rotate_filter(straight1,2)
straight5 = rotate_filter(straight1,3)
straight6 = rotate_filter(straight2,1)


#Diagonal Filters
diagonal1 = np.array([[1,-1,-1],
                      [-1,1,-1],
                      [-1,-1,1]])
diagonal2 = rotate_filter(diagonal1,1)

#Corner Filters
corner1 = np.array([[-1,1,-1],
                    [1,1,-1],
                    [-1,-1,-1]])
corner2 = rotate_filter(corner1,1)
corner3 = rotate_filter(corner1,2)
corner4 = rotate_filter(corner1,3)



filters = [straight1,straight2,straight3,straight4,straight5,straight6,diagonal1,diagonal2,corner1,corner2,corner3,corner4]
# 12 filters

def process(size,img):
    global filters
    (width,height) = size
    layer = []
    for a in range(len(filters)):
        slot = []
        for x in range(height):
            for z in range(width):
                sliced = np.array(get_portion(img,(z,x)))
                out = get_convolutions(filters[a],sliced)
                slot.append(out)
        
        layer.append(np.reshape(slot,(-1,width)))
    return layer




