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

'''
This code is used to plot the performance of the Gaussian Process
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

#Code used to reduce the length of each column on the images
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
    return finalMatrix, randomIndexes, len(listing)



if __name__ == "__main__":
    #NumImages = Trainingsize + 1 
    numImages = 11
    tic = time.perf_counter()
    finalComponents, randIndex, lenX = PCAImagesCompression("/home/chen2156/laserData/src/laser_values/src/newTrainingImages/unWarpedImages/", numImages)
    toc = time.perf_counter()
    diffTime = toc - tic
    print("Time taken to run  PCA is " + str(diffTime) + " seconds")
    ObservedData = np.loadtxt("/home/chen2156/laserData/src/laser_values/src/laserDataCaputer.csv", delimiter=',')
    x = np.atleast_2d(np.linspace(0, 360, 100)).T
    kernel = RBF()
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=100)
    leny = len(ObservedData)

   
    #y = np.tile(ObservedData, numImages - 1)
    y = np.array([])
    yIndexes = []
    for ind in range(len(randIndex)):
        yInd  = round(leny / lenX * randIndex[ind])
        yIndexes.append(yInd)
        y = np.append(y, ObservedData[yInd, :])

    X = finalComponents[0:360 * (numImages - 1), :]
    x = finalComponents[360 * (numImages - 1):360 * numImages, :]
    tic = time.perf_counter()
    y = y[0:360 * (numImages - 1)]

    print(X.shape)
    print(y.shape)
    print(x.shape)
    gp.fit(X, y)
    toc = time.perf_counter()
    diffTime = toc - tic
    print("Time taken to run  Gaussian Process Train is " + str(diffTime) + " seconds")

    tic = time.perf_counter()
    y_pred, sigma = gp.predict(x, return_std=True) 
    toc = time.perf_counter()
    diffTime = toc - tic
    print("Time taken to run  Gaussian Process Predict is " + str(diffTime) + " seconds")

    plt.figure()
    plt.plot(range(360), ObservedData[yIndexes[-1]], "r.", markersize=10, label="Observations")
    plt.plot(range(360), y_pred, "b-", label="Prediction")
    plt.fill(
        np.concatenate([range(360), range(360)[::-1]]),
        np.concatenate([y_pred - 1.9600 * sigma, (y_pred + 1.9600 * sigma)[::-1]]),
        alpha=0.5,
        fc="b",
        ec="None",
        label="95% confidence interval",
    )
    plt.xlabel("Degree ")
    plt.ylabel("Distance (m)")
    plt.ylim(-10, 20)
    plt.legend(loc="upper left")
    plt.savefig("/home/chen2156/laserData/src/laser_values/src/newTrainingImages/10TrainingImages.png")
    
    

