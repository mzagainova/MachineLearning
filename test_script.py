import numpy as np
from perceptron import *

#X = np.array([[0,1],[1,0],[5,4],[1,1],[3,3],[2,4],[1,6]])
#Y = np.array([[1],[1],[-1],[1],[-1],[-1],[-1]])

X = np.array([[-2,1],[1,1],[1.5,-0.5],[-2,-1],[-1,-1.5],[2,-2]])
Y = np.array([[1],[1],[1],[-1],[-1],[-1]])
W = perceptron_train(X,Y)
test_acc = perceptron_test(X,Y,W[0],W[1])
print W
print test_acc
