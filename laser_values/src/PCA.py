import numpy as np
import cv2

def PCA(filepath):
    image = cv2.imread(filepath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


    #Standardize/Normalize the image

    H = image[:, :, 0]
    S = image[:, :, 1]
    V = image[:, :, 2]

    meanH = H.mean()
    stdH = H.std()

    meanS = S.mean()
    stdS = S.std()

    meanV = V.mean()
    stdV = V.std()

    normH = (H - meanH) / stdH 
    normS = (S - meanS) / stdS
    normV = (V - meanV) / stdV

    #Compute covariance Matrix
    reshapeH = normH.flatten()
    reshapeS = normS.flatten()
    reshapeV = normV.flatten()

    mat = np.vstack((reshapeH, reshapeS, reshapeV))
    covMatrix = np.cov(mat)

    #Compute Eigenvalues and Eigenvectors

    eigVal, eigVec = np.linalg.eig(covMatrix)
    print(eigVec)

    #Rank eigenvectors based on eigenvalues's value

    minIndex = np.argmin(eigVal)
    maxIndex = np.argmax(eigVal)
    midIndex = np.where(eigVal == np.median(eigVal))
    midIndex = midIndex[0][0]

    print(minIndex, midIndex, maxIndex)


    minEigenVector = eigVec[minIndex]
    midEigenVector = eigVec[midIndex]
    maxEigenVector = eigVec[maxIndex]

    percentMaxEig = eigVal[maxIndex] / sum(eigVal)
    percentMidEig = eigVal[midIndex] / sum(eigVal)
    percentMinEig = eigVal[minIndex] / sum(eigVal)

    print(percentMaxEig)
    print(percentMidEig)
    print(percentMinEig)

    #Create Feature Vector

    FeatureVector = np.column_stack((eigVec[0], eigVec[1], eigVec[2]))

    print(FeatureVector)

    #Recast data along the PCA

    FinalDataset = FeatureVector.T * image.T

    return FinalDataset







if __name__ == "__main__":
    filename = "/home/chen2156/laserData/src/laser_values/src/datapoint3/image.png"
    res = PCA(filename)
    print(res)
