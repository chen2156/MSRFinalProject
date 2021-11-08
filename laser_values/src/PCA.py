import numpy as np
import cv2
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.stats import stats
import matplotlib.image as mpimg
from glob import iglob

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C


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
    flattenArraysH = pd.DataFrame([])
    flattenArraysS = pd.DataFrame([])
    flattenArraysV = pd.DataFrame([])
    listing = os.listdir(fileDirectory)
    i = 0  
    for file in listing:
        if i == 600:
            break
        image = cv2.imread(fileDirectory + file)
        print(image.shape)
        resized = cv2.resize(image, (1376, 219), interpolation=cv2.INTER_LINEAR)
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
        
        dfH = pd.Series(scaledH.flatten(), name=file)
        flattenArraysH = flattenArraysH.append(dfH)
        dfS = pd.Series(scaledS.flatten(), name=file)
        flattenArraysS =flattenArraysS.append(dfS)
        dfV = pd.Series(scaledV.flatten(), name=file)
        flattenArraysV = flattenArraysV.append(dfV)
        i += 1


    #print(flattenArraysH)
    #print(flattenArraysS)
    #print(flattenArraysV)

    pcaH = PCA(n_components=6)
    pcaH.fit(flattenArraysH)
    

    pcaS = PCA(n_components=6)
    pcaS.fit(flattenArraysS)

    pcaV = PCA(n_components=6)
    pcaV.fit(flattenArraysV)



    transPCAH = pcaH.transform(flattenArraysH)
    transPCAS = pcaS.transform(flattenArraysS)
    transPCAV = pcaV.transform(flattenArraysV)

    print(transPCAH.shape)
    print(transPCAS.shape)
    print(transPCAV.shape)

    print(f"Hue Channel : {sum(pcaH.explained_variance_ratio_)}")
    print(f"Saturation Channel: {sum(pcaS.explained_variance_ratio_)}")
    print(f"Variation Channel  : {sum(pcaV.explained_variance_ratio_)}")

    projectH = pcaH.inverse_transform(transPCAH)
    print("Done with PCAH")
    projectS = pcaS.inverse_transform(transPCAS)
    print("Done with PCAS")
    projectV = pcaV.inverse_transform(transPCAV)
    print("Done with PCAV")

    print("Done running PCAs")

    return [transPCAH, transPCAS, transPCAV, projectH, projectS, projectV, sum(pcaH.explained_variance_ratio_), sum(pcaS.explained_variance_ratio_), sum(pcaV.explained_variance_ratio_)]



if __name__ == "__main__":
    print("Start the processs")
    [finalH, finalS, finalV, projH, projS, projV, varH, varS, varV] = PCAImagesCompression("/home/chen2156/laserData/src/laser_values/src/multipleImages/unWarpedImages/")
    print("Done with transforming shape")
    print("Final Transformation shape is")
    print(finalH.shape)
    print(finalS.shape)
    print(finalV.shape)

    hsv_image = cv2.merge([projH, projS, projV])

    print(projH[0].shape)
    print(projS[0].shape)
    print(projV[0].shape)

    H = projH[0].reshape(219, 1376)
    S = projS[0].reshape(219, 1376)
    V = projV[0].reshape(219, 1376)

    hsvImage = cv2.merge([H, S, V])
    print(hsvImage.shape)
    listing = os.listdir("/home/chen2156/laserData/src/laser_values/src/multipleImages/unWarpedImages/")  
    origImage = cv2.imread("/home/chen2156/laserData/src/laser_values/src/multipleImages/unWarpedImages/" + listing[0], cv2.COLOR_BGR2HSV)
    print("Starting Gaussian Process")

    #kernel = C(1.0, (1e-3, 1e3)) * RBF([5,5], (1e-2, 1e2))

    if max([varH, varS, varV]) == varH:
        X = projH[0]
    elif max([varH, varS, varV]) == varS:
        X = projS[0]
    else:
        X = projV[0]

    X = cv2.resize(X, (360, 219), interpolation=cv2.INTER_LINEAR)  
    print(X.shape)  

    kernel = RBF(length_scale=[1./X.shape[1] for i in range(X.shape[1])])


    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=15)

    f = open("/home/chen2156/laserData/src/laser_values/src/multipleImages/laserDataCaputer.txt", "r")
    
    line = f.readline()

    y  = line.split(",")
    y = list(map(float, y))
    

    gp.fit(X, y)
    x = range(360)
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
    
    
    

