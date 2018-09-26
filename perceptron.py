import numpy as np

X = np.array([[-2,1],[1,1],[1.5,-0.5],[-2,-1],[-1,-1.5],[2,-2]])
Y = np.array([[1],[1],[0],[1],[0],[0],[0]])

def update(W,X,Y,b,sample):
    for feat in range(X.shape[1]):
        W[feat] = W[feat] + X[sample][feat] * Y[sample]
    print "W!!!\n"
    print W
    b = b + Y[sample]

def perceptron_train(X,Y):
    flag = True
    W = []
    b = 0
    for i in range(X.shape[1]):
        W.append(0)
    # until goes through one epoch without updating
    #while flag:
    for k in range(0,5):
        print "EPOCH !!!!\n\n\n"
        flag = False
        # for every sample, calculate a
        for sample in range(X.shape[0]):
            # for every feature Xi
            a = 0
            for feat in range(X.shape[1]):
                a += W[feat] * X[sample][feat]
                print a
            a += b
            print a
            if a <= 0:
                flag = True
                update(W,X,Y,b,sample)
    result = []
    #result.append[W]
    print W
    #result.append[b]
    print b
    return result

def perceptron_test(X_test,Y_test,w,b):
    return 0

perceptron_train(X,Y)
