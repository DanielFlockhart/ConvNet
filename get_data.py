import cv2,random
import numpy as np

def ran(lim,hig):
    return random.randint(lim,hig)


def triangle(name):
    image = np.ones((64, 64, 3), np.uint8) * 255
    pt1 = (ran(12,54), ran(12,54))
    pt2 = (ran(12,54), ran(12,54))
    pt3 = (ran(12,54), ran(12,54))
    shape = np.array( [pt1, pt2, pt3] )
    cv2.drawContours(image, [shape], 0, get_colour(), -1)
    cv2.imwrite(f'data/triangle{name}.png',image)

def square(name):
    image = np.ones((64, 64, 3), np.uint8) * 255
    x1 = ran(2,30)
    x2 = x1+ran(2,30)
    y1 = ran(2,30)
    y2 = y1 + ran(2,30)
    cv2.rectangle(image,(x1,y1),(x2,y2),get_colour(),-1)
    cv2.imwrite(f'data/square{name}.png',image)
    
def circle(name):
    image = np.ones((64, 64, 3), np.uint8) * 255
    x1 = ran(16,48)
    y2 = ran(16,48)
    rad = ran(8,24)
    cv2.circle(image,(x1,y2),rad,get_colour(), -1)
    cv2.imwrite(f'data/circle{name}.png',image)

    
def get_colour():
    return (0,0,0)#(ran(0,255),ran(0,255),ran(0,255))

def run():
    for x in range(100):
        triangle(x)
        square(x)
        circle(x)
        
if __name__ == "__main__":
    run()
