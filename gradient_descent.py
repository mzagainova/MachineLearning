import numpy as np

def df(x):
    return 2*x

def checkAll(x):
    if x.shape[0] > 1:
        for i in range(x.shape[0]):
            if x[i] >= 0.0001:
                return False
    elif x >= 0.0001:
        return False
    return True

def gradient_descent(f, x_init, n):
    y = 0
    while(not checkAll(x_init)):
        x_init -= n*(f(x_init))
        y +=1
    print y
    return x_init

x = gradient_descent(df, np.array([5.0]),0.1)
print x

x = gradient_descent(df, np.array([5.0,5.0]),0.1)
print x
