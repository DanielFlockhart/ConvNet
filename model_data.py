import os,ast

def save_model(name,value):
    with open("Models//"+name+"weights.txt",'w') as file:
        file.write(str(value[0]))
        
    with open("Models//"+name+"biases.txt",'w') as file:
        file.write(str(value[1]))    


def load_model(name):
    weights = []
    biases = []
    with open("Models//"+name+"weights.txt","r") as file:
        weights = literal(file.read())
    with open("Models//"+name+"biases.txt","r") as file:
        biases = literal(file.read())
    return (weights,biases)

def literal(model):
    return ast.literal_eval(model)
