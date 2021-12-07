import numpy as np
import cv2
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.stats import stats
import matplotlib.image as mpimg
from glob import iglob
import random

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.gaussian_process.kernels import _check_length_scale
from sklearn import preprocessing
import time

import joblib
import pickle

'''
This script is used to train the model and save it into a .sav file
'''

def PCAImageCompression(imageFile):
    image = cv2.cvtColor(cv2.imread(imageFile), cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(image)

    pca = PCA(6)
    hTransformed = pca.fit_transform(H)
    hInverted = pca.inverse_transform(hTransformed)

    sTransformed = pca.fit_transform(S)
    sInverted = pca.inverse_transform(sTransformed)

    vTransformed = pca.fit_transform(V)
    vInverted = pca.inverse_transform(vTransformed)

    imageCompressed = (np.dstack((hInverted, sInverted, vInverted))).astype(np.uint8)

    return imageCompressed

def PCAImagesCompression(fileDirectory, numImages):
    
    listing = os.listdir(fileDirectory)
    randomIndexes = random.sample(range(0, len(listing)), numImages)


    flattenedMatrix = np.empty((420, 360))
    for i in range(numImages):
        image = cv2.imread(fileDirectory + listing[randomIndexes[i]])
        resized = cv2.resize(image, (360, 140), interpolation=cv2.INTER_LINEAR)
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        H, S, V = cv2.split(resized)
        scaledH = H / 180
        scaledS = S / 255
        scaledV = V / 255
        flattenedImage = np.vstack((scaledH, scaledS, scaledV))
        if i == 0:
            flattenedMatrix[:] = flattenedImage
        else:
            flattenedMatrix = np.hstack((flattenedMatrix, flattenedImage))    

    finalMatrix = np.empty((6, 1))
    pca = PCA(6)

    pca.fit(flattenedMatrix.T)

    finalMatrix = pca.transform(flattenedMatrix.T)
    return finalMatrix

if __name__ == "__main__":
    #NumImages = Trainingsize + 1 
    numImages = 6
    tic = time.perf_counter()
    finalComponents = PCAImagesCompression("/home/chen2156/laserData/src/laser_values/src/multipleImages/unWarpedImages/", numImages)
    toc = time.perf_counter()
    diffTime = toc - tic
    print("Time taken to run  PCA is " + str(diffTime) + " seconds")
    ObservedData = np.loadtxt("/home/chen2156/laserData/src/laser_values/src/multipleImages/laserDataCaputer.csv", delimiter=',')
    x = np.atleast_2d(np.linspace(0, 360, 100)).T
    kernel = RBF()
    y = np.tile(ObservedData, numImages - 1)
    X = finalComponents[0:360 * (numImages - 1), :]
    x = finalComponents[360 * (numImages - 1):360 * numImages, :]

    print(X.shape)
    print(y.shape)

    print("Size of the array X: ",
      X.size)
  
    print("Memory size of one array element in bytes: ",
      X.itemsize)
  
    # memory size of numpy array
    print("Memory size of numpy array X in bytes:",
      X.size * X.itemsize)

    print("Size of the array y: ",
      y.size)
  
    print("Memory size of one array element in bytes: ",
      y.itemsize)
  
    # memory size of numpy array
    print("Memory size of numpy array y in bytes:",
      y.size * y.itemsize)

    TotalMemorySize = X.size * X.itemsize + y.size * y.itemsize
    print("Total memory size in bytes is: " + str(TotalMemorySize))

    memGB = TotalMemorySize / 1e9

    print("Total memory size in GB is: " + str(memGB))

    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=100, copy_X_train=False)
    tic = time.perf_counter()
    gp.fit(X, y)
    toc = time.perf_counter()
    diffTime = toc - tic
    print("Time taken to train GP is " + str(diffTime) + " seconds")
    filename = "/home/chen2156/laserData/src/laser_values/src/5ImagesGaussianProcessModel.sav"
    joblib.dump(gp, filename)

   
