import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as la
from decimal import Decimal
import os as os

# X = np.array([[-1, -1], [-1,1], [1, -1], [1,1]])

def compute_Z(X, centering=True, scaling=False):
    X = np.array(X)
    X = X.astype(float)
    if centering:
        for col in range(X.shape[1]):
            mean = float(np.mean(X[:,col]))
            if scaling:
                std = float(np.std(X[:,col]))
            for row in range(X.shape[0]):
                X[row,col] = float(X[row,col]) - float(mean)
                if scaling:
                    X[row,col] = float(X[row,col]) / float(std)
    return X

def compute_covariance_matrix(Z):
    return Z.transpose().dot(Z)

def find_pcs(COV):
    eigValues, eigVector = la.eig(COV)
    zipped = zip(eigValues, eigVector)
    array = zip(*sorted(zip(eigValues, eigVector), key=lambda tup: tup[0], reverse=True))
    return np.array(array[0]), np.array(array[1])

def project_data(Z, PCS, L, k, var):
    if var > 0:
        k = 0
        sumVars = 0
        vars = np.array(L) / sum(L)
        for i in vars:
            k += 1
            sumVars += i
            if(sumVars >= var):
                break
    if k > 0:
        Z_star = Z.dot(PCS[:,0:k])
    return Z_star


# #print X
# Z = compute_Z(X, True, True)
# #print Z
#
# Zcov = compute_covariance_matrix(Z)
# #print Zcov
#
# eigValues, eigVector = find_pcs(Zcov)
# print eigValues
# Z_star = project_data(Zcov, eigVector, eigValues, 0, 1)
# print Z_star

# Z = compute_Z(X)
# COV = compute_covariance_matrix(Z)
# print COV
# L, PCS = find_pcs(COV)
# Z_star = project_data(Z ,PCS, L,1,0)
# print Z_star
