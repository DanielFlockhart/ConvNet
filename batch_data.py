import random,time



def get_image(shapes,batch_size,image_count):
    shape = shapes[random.randint(0,len(shapes)-1)]
    index = random.randint(0,image_count-1)
    image_name = f"{shape}{index}"
    return image_name
def get_batch(batch_size,folder_size,shapes):
    images = []
    for x in range(batch_size):
        images.append(get_image(shapes,batch_size,folder_size))
    return images
