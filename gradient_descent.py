import numpy as np

'''
def df(x):
    return 2*x

def df2(x):
    return np.array(2*x[0], 2*x[1])

def df3(x):
    return np.array([4*math.pow(x[0],3), 6*x[1]])
'''

def checkAll(x):
    if x.shape[0] > 1:
        for i in range(x.shape[0]):
            if x[i] >= 0.0001:
                return False
    elif x >= 0.0001:
        return False
    return True

def gradient_descent(f, x_init, n):
    grad = f(x_init)
    while(not checkAll(grad)):
        x_init -= n*(f(x_init))
        grad = f(x_init)
    return x_init
