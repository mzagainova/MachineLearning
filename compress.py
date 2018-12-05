import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as la
from decimal import Decimal
import os as os
import pca as pca

fileNames = []

def load_data(input_dir):
    global fileNames
    os.chdir(input_dir)
    data = []
    for file in os.listdir(os.getcwd()):
        if file == '.DS_Store':
            continue
        fileNames.append(file)
        data.append(np.array(plt.imread(file)).astype(float).flatten())
    return np.asarray(data).transpose().astype(float)

def scale(Z):
    for i in range(0,Z.shape[1]):
        min = np.abs(Z[:, i].min())
        max = Z[:, i].max()
        max = float(255) / float(max)
        for j in range(0, Z.shape[0]):
            Z[j][i] += min
            Z[j][i] *= max
    return Z

def compress_images(DATA,k):
    global fileNames
    Z = pca.compute_Z(DATA)
    Z = Z.astype(float)
    COV = pca.compute_covariance_matrix(Z)
    L, PCS = pca.find_pcs(COV)
    Z_star = pca.project_data(Z ,PCS, L,k,0)
    # Xcompressed = Z * U^T
    Z_star = Z_star.dot(PCS[:,0:k].transpose())
    Z_star = Z_star.astype(float)
    Z_star = scale(Z_star)
    if not os.path.isdir(os.path.join(os.getcwd(), r'Output')):
        os.mkdir(os.path.join(os.getcwd(), r'Output'))
    for j in range(0, Z_star.shape[1]):
        imgData = []
        for i in range(0,Z_star.shape[0]):
            imgData.append(Z_star[i][j])
        imgData = np.asarray(imgData)
        imgData = imgData.reshape(60,48)
        filename = os.path.join(os.getcwd(), r'Output/')+str(fileNames[j].strip('.pgm'))+".png"
        fileDir = os.path.expanduser(filename)
        plt.imsave(fileDir, imgData, cmap='gray')

# X = load_data('Data/Test/')
# compress_images(X, 10)
