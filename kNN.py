import numpy as np
import math

X_test = np.array([[1,1], [2,1], [0,10], [10,10], [5,5], [3,10], [9,4], [6,2], [2,2], [8,7]])
Y_test = np.array([1, -1, 1, -1, 1, -1, 1, -1, 1, -1])
X_train = np.array([[1,5], [2,6], [2,7], [3,7], [3,8], [4,8], [5,1], [5,9], [6,2], [7,2], [7,3], [8,3], [8,4], [9,5]])
Y_train = np.array([-1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, 1])
K = 5


def calcDistance(X_train, X_test):
    distance = 0.00
    for i in range(0, X_train.shape[0]):
        distance += (X_train[i] - X_test[i]) ** 2
    return distance

def KNN_test(X_train, Y_train, X_test, Y_test, K):
    correct = 0
    # compute label for each test sample
    for i in range(0, X_test.shape[0]):
        distances = []
        # calculate distance between sample and all training points
        for j in range(0, X_train.shape[0]):
            distances.append(calcDistance(X_train[j], X_test[i]))
        # sort distances while preserving indeces, distIndex holds indeces of sorted distances
        distIndex = sorted(range(len(distances)),key=distances.__getitem__)
        print distIndex
        # sum labels of closest K labels
        kSum = 0
        for k in range(0, K):
            kSum += Y_train[distIndex[k]]
        # check if our label is correct
        if kSum == Y_test[i]:
            correct += 1
    #calculate and return accuracy
    return float(correct) / float(Y_test.shape[0])

accuracy = KNN_test(X_train, Y_train, X_test, Y_test, K)
print accuracy
