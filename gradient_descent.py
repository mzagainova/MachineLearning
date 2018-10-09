import numpy as np

def df(x):
    return 2*x

def gradient_descent(f, x_init, n):
    y = 0
    while(x_init >= 0.0001):
        x_init -= n*(f(x_init))
        y +=1
    print y
    return x_init

x = gradient_descent(df, np.array([5.0]),0.1)
print x
