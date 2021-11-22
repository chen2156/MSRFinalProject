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

def PCAImagesCompression(fileDirectory):
    
    listing = os.listdir(fileDirectory)
    numImages = 1
    randomIndexes = random.sample(range(0, len(listing)), numImages)
    print(randomIndexes)

    flattenedMatrix = np.empty((657, 360))
    for i in range(numImages):
        image = cv2.imread(fileDirectory + listing[randomIndexes[i]])
        print(image.shape)
        resized = cv2.resize(image, (360, 219), interpolation=cv2.INTER_LINEAR)
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        print(resized.shape)
        print(resized.size)

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

    pca6 = PCA(n_components=6)
    for i in range(flattenedMatrix.shape[1]):
        if i == 10:
            break    
        reshap = flattenedMatrix[:, i].reshape(-1, 1)
        print("reshape's shape is")
        print(reshap.shape)
        covMatrix = np.cov(reshap)
        print("Covariance Matrix shape is")
        print(covMatrix.shape)
        covMatrix = np.nan_to_num(covMatrix)
        eigenValues, eigenVectors = np.linalg.eig(covMatrix)

        idx = eigenValues.argsort()[::-1]   
        eigenValues = eigenValues[idx]
        eigenVectors = eigenVectors[:,idx]

        featureVector = eigenVectors[:, len(eigenVectors) - 6: len(eigenVectors)]

        print("shape of featureVector is")
        print(featureVector.shape)

        finalTransform = np.matmul(featureVector.T, reshap)

        print("shape of finalTransform is")
        print(finalTransform.shape)


        if i == 0:
            finalMatrix[:] = finalTransform
            print("shape of finalMatrix is")
            print(finalMatrix.shape)

            print(finalMatrix.shape)
        else:
            finalMatrix = np.hstack((finalMatrix, finalTransform))
            print("shape of finalMatrix is")
            print(finalMatrix.shape)
    print(finalMatrix.shape)

    return finalMatrix



if __name__ == "__main__":
    print("Start the processs")
    finalComponents = PCAImagesCompression("/home/chen2156/laserData/src/laser_values/src/multipleImages/unWarpedImages/")
    print("Shape of final Component is")
    print(finalComponents.shape)

    ObservedData = np.loadtxt("/home/chen2156/laserData/src/laser_values/src/multipleImages/laserDataCaputer.csv", delimiter=',')

    x = np.atleast_2d(np.linspace(0, 360, 1000)).T

    kernel = RBF()
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=100)

    ypredResult = []
    MSEResult = []
    for c in range(finalComponents.shape[1]):

        X = finalComponents[:, c].reshape(-1, 1)

        print(X)
        y = np.array([ObservedData[c % 360]]).reshape(1, -1)

        print(y)

        print("Shapes are")
        print(X.shape)
        print(y.shape)
        gp.fit(X.T, y)
        x = c % 360
        y_pred, MSE = gp.predict(x, return_std=True)
        ypredResult.append(y_pred)
        MSEResult.append(MSE)

    print("Results are")
    print(ypredResult)
    print(MSEResult)


    #kernel = C(1.0, (1e-3, 1e3)) * RBF([5,5], (1e-2, 1e2))

    #if max([varH, varS, varV]) == varH:
    #    X = projH[0]
    #elif max([varH, varS, varV]) == varS:
    ##    X = projS[0]
    #else:
    #    X = projV[0]
    #kernel = RBF(length_scale=[1./X.shape[0] for i in range(X.shape[0])])
    #kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))



    # f = open("/home/chen2156/laserData/src/laser_values/src/multipleImages/laserDataCaputer.txt", "r")
    
    #line = f.readline()

    #y  = line.split(",")
    #y = np.array(list(map(float, y)))
    #print("Dimensions are")

    
  

    
    #x = np.array(range(360)).reshape(1, -1)
    #print("Shape is")
    #print(x.shape)
    
    
 
    
    plt.figure()
    #plt.plot(x, f(x), "r:", label=r"$f(x) = x\,\sin(x)$")
    plt.plot(X, y, "r.", markersize=10, label="Observations")
    plt.plot(x, y_pred, "b-", label="Prediction")
    plt.fill(
        np.concatenate([x, x[::-1]]),
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
    plt.show()
    
    
    

