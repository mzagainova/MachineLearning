import numpy as np

X = np.array([[-2,1],[1,1],[1.5,-0.5],[-2,-1],[-1,-1.5],[2,-2]])
#X = np.array([[0,1],[1,0],[5,4],[1,1],[3,3],[2,4],[1,6]])
#Y = np.array([[1],[1],[-1],[1],[-1],[-1],[-1]])
Y = np.array([[1],[1],[1],[-1],[-1],[-1]])

def update(W,X,Y,b,sample):
    for feat in range(X.shape[1]):
        W[feat] = W[feat] + X[sample][feat] * Y[sample]
    b = b + Y[sample]
    return b

def perceptron_train(X,Y):
    flag = True
    W = np.zeros(X.shape[1])
    b = 0
    # until goes through one epoch without updating
    while flag:
        flag = False
        # for every sample, calculate a
        for sample in range(X.shape[0]):
            # for every feature Xi
            a = 0
            for feat in range(X.shape[1]):
                a += W[feat] * X[sample][feat]
            a += b
            if (a * Y[sample]) <= 0:
                flag = True
                b = update(W,X,Y,b,sample)
    result = []
    result.append(W)
    result.append(b)
    return result

def perceptron_test(X_test,Y_test,W,b):
    correct = 0
    for sample in range(X_test.shape[0]):
        a = 0
        for feat in range(X_test.shape[1]):
            a += (X_test[sample][feat] * W[feat])
        a += b
        if a > 0:
            if Y_test[sample] == 1:
                correct += 1
        else:
            if Y_test[sample] == -1:
                correct += 1
    return float(correct) / float(Y_test.shape[0])

answer = perceptron_train(X,Y)
print answer[0]
print answer[1]
print perceptron_test(X,Y,answer[0],answer[1])
