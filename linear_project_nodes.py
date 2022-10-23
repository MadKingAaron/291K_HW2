
initial_size = 32

layers = [{'s':1, 'k':5, 'p':0}, {'s':2, 'k':2, 'p':0}, {'s':1, 'k':5, 'p':0}, {'s':2, 'k':2, 'p':0}, {'s':1, 'k':5, 'p':1}, {'s':1, 'k':2, 'p':0}]



def size(w, k, p, s):
    return ((w-k+2*p)/s)+1


def get_final_size(layers, w):
    for layer in layers:
        w = size(w, layer['k'], layer['p'], layer['s'])
    return w


print(get_final_size(layers, initial_size))