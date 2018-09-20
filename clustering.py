import numpy as np
import math
import random as rand

X  =  np.array([[0], [1], [2], [7], [8], [9], [12], [14], [15]])
K = 3

def calcDistance(X1, X2):
    distance = 0.00
    for i in range(0, X2.shape[0]):
        distance += (X1[i] - X2[i]) ** 2
    return distance

def K_Means(X, K):
    centers = rand.sample(range(min(X), max(X), K))
    clusters = [K][]
    minDist = sys.maxint
    # until convergence reached
    while :
        for i in range(0, X):
            for j in range(0, K):
                dist = calcDistance(X[i], centers[j])
                if dist < minDist:
                    c = j
            clusters[c].append(X[i])
            
