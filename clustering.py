import numpy as np
import math
import random as rand
import matplotlib.pyplot as plt
import sys

#X1  =  np.array([ [0],[1],[2],[7] ,[ 8 ] , [ 9 ] , [ 12 ] , [ 14 ] , [ 15 ] ] )
#X2 =  np.array([[0,1], [1,3], [2,1], [7,3], [8,4], [9,1], [12,0], [14,2], [15,3]])
X1 = np.array([[1,0],[7,4],[9,6],[2,1],[4,8],[0,3],[13,5],[6,8],[7,3],[3,6],[2,1],[8,3],[10,2],[3,5],[5,1],[1,9],[10,3],[4,1],[6,6],[2,2]])

def calcDistance(X1, X2):
    distance = 0.00
    for i in range(0, len(X2)):
        distance += (X1[i] - X2[i]) ** 2
    return distance

def recalcCenters(X, Y):
    cSize = len(X)
    newCenter = []
    if len(X) == 0:
        return Y
    for j in range(len(X[0])):
        sum = 0
        for i in range(len(X)):
            sum += X[i][j]
        newCenter.append(float(sum) / float(cSize))
    return newCenter

def checkExisting(X, clusters):
    for i in range(len(clusters)):
            if (X == clusters[i]).all():
                return True
    return False

def K_Means(X, K):
    X.tolist()
    centerIndices = rand.sample(range(0, len(X)), K)
    centers = []
    for i in range(len(centerIndices)):
        centers.append(X[centerIndices[i]])
    #print "Starting centers:"
    #print centers
    flag = True
    # until convergence reached
    while flag:
        oldCenters = centers
        clusters = []
        for i in range(K):
            clusters.append([])
        flag = False
        for i in range(0, len(X)):
            minDist = sys.maxint
            # calc distance from point to every cluster center
            for j in range(0, K):
                dist = calcDistance(X[i], centers[j])
                if dist < minDist:
                    minDist = dist
                    c = j
            if not checkExisting(X[i],clusters[c]):
                clusters[c].append(X[i])
                flag = True
        for k in range(0,K):
            centers[k] = recalcCenters(clusters[k], centers[k])
        if centers == oldCenters:
            flag = False
    for i in range(K):
        #print "Cluster " + str(i)
        #print clusters[i]
        for j in range(len(clusters[i])):
            point = clusters[i][j]
            if i == 0:
                c = 'r'
            elif i == 1:
                c = 'b'
            else:
                c = 'g'
            plt.scatter(point[0],point[1],c=c,s=50)
    #print "centers"
    for j in range(len(centers)):
        point = centers[j]
        if j == 0:
            c = 'r'
        elif j == 1:
            c = 'b'
        else:
            c = 'g'
        plt.scatter(point[0],point[1],c=c,s=80)
    #plt.show()
    return centers

'''K = 2
print "K = 2"
print K
print K_Means(X1, K)

print "K = 3"
K = 3
print K
print K_Means(X1, K)'''
