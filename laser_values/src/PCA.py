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
    numImages = 7
    randomIndexes = random.sample(range(0, len(listing)), numImages)
    flattenArraysH = np.empty((78840, numImages))
    flattenArraysS = np.empty((78840, numImages))
    flattenArraysV = np.empty((78840, numImages))
    for i in range(numImages):
        image = cv2.imread(fileDirectory + listing[randomIndexes[i]])
        print(image.shape)
        resized = cv2.resize(image, (360, 219), interpolation=cv2.INTER_LINEAR)
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        print(resized.shape)
        print(resized.size)

        H, S, V = cv2.split(resized)

        #print(H.shape)
        #print(S.shape)
        #print(V.shape)

        scaledH = H / 180
        scaledS = S / 255
        scaledV = V / 255
        
        #dfH = pd.Series(scaledH.flatten(), name=file)
        #flattenArraysH = flattenArraysH.append(dfH)
        #dfS = pd.Series(scaledS.flatten(), name=file)
        #flattenArraysS =flattenArraysS.append(dfS)
        #dfV = pd.Series(scaledV.flatten(), name=file)
        #flattenArraysV = flattenArraysV.append(dfV)

        flattenArraysH[:, i] = scaledH.flatten()
        flattenArraysS[:, i] = scaledS.flatten()
        flattenArraysV[:, i] = scaledV.flatten()





    #print(flattenArraysH)
    #print(flattenArraysS)
    #print(flattenArraysV)

    #pcaH = PCA(n_components=6)
    #pcaH.fit(flattenArraysH)
    

    #pcaS = PCA(n_components=6)
    #pcaS.fit(flattenArraysS)

    #pcaV = PCA(n_components=6)
    #pcaV.fit(flattenArraysV)

    finalMatrix = np.vstack((flattenArraysH, flattenArraysS, flattenArraysV))

    pca6 = PCA(n_components=6)
    pca6.fit(finalMatrix)
    transformedImages = pca6.transform(finalMatrix)

    projectedImages = pca6.inverse_transform(transformedImages)

    #transPCAH = pcaH.transform(flattenArraysH)
    #transPCAS = pcaS.transform(flattenArraysS)
    #transPCAV = pcaV.transform(flattenArraysV)

    #print(transPCAH.shape)
    #print(transPCAS.shape)
    #print(transPCAV.shape)

    #print(f"Hue Channel : {sum(pcaH.explained_variance_ratio_)}")
    #print(f"Saturation Channel: {sum(pcaS.explained_variance_ratio_)}")
    #print(f"Variation Channel  : {sum(pcaV.explained_variance_ratio_)}")

    #projectH = pcaH.inverse_transform(transPCAH)
    #print("Done with PCAH")
    #projectS = pcaS.inverse_transform(transPCAS)
    #print("Done with PCAS")
    #projectV = pcaV.inverse_transform(transPCAV)
    #print("Done with PCAV")

    #print("Done running PCAs")

    return projectedImages, transformedImages



if __name__ == "__main__":
    print("Start the processs")
    projectedImages, transformedImages = PCAImagesCompression("/home/chen2156/laserData/src/laser_values/src/multipleImages/unWarpedImages/")
    print("Done with transforming shape")
    print("Final Transformation shape is")

    X = projectedImages[:, 0].reshape(657, 360)
    print("Shape of X is")
    print(X.shape)
    #kernel = C(1.0, (1e-3, 1e3)) * RBF([5,5], (1e-2, 1e2))

    #if max([varH, varS, varV]) == varH:
    #    X = projH[0]
    #elif max([varH, varS, varV]) == varS:
    ##    X = projS[0]
    #else:
    #    X = projV[0]
    #kernel = RBF(length_scale=[1./X.shape[0] for i in range(X.shape[0])])
    kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))


    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=30)

    # f = open("/home/chen2156/laserData/src/laser_values/src/multipleImages/laserDataCaputer.txt", "r")
    
    #line = f.readline()

    #y  = line.split(",")
    #y = np.array(list(map(float, y)))
    #print("Dimensions are")

    y = np.loadtxt("/home/chen2156/laserData/src/laser_values/src/multipleImages/laserDataCaputer.csv", delimiter=',')
    
    print("X and y dimensions are")

    print(X.shape)
    print(y.shape)

    gp.fit(X.T, y)


    x = np.atleast_2d(np.linspace(0, 360, 1000)).T
    print(x.shape)
    #x = np.array(range(360)).reshape(1, -1)
    #print("Shape is")
    #print(x.shape)
    y_pred, MSE = gp.predict(x, return_std=True)
    
    print("Results are")
    print(y_pred.shape)
    print(MSE)
    
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
    
    
    

