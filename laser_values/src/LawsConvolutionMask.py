import numpy as np
import cv2

def LawsConvo(filename, lightMapDamping):
    image = cv2.imread(filename)

    #Convert Image to Grayscale
    origImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #Gaussian Smoothing
    gaussImage = cv2.GaussianBlur(image,(5,5),cv2.BORDER_DEFAULT)

    
    print(image)

    if lightMapDamping:
        copyimage = cv2.imread(filename)
        copyimage = cv2.cvtColor(copyimage, cv2.COLOR_BGR2GRAY)
        copyimage = cv2.GaussianBlur(copyimage,(5,5),cv2.BORDER_DEFAULT)
        image = image * copyimage

    # show the output image
    cv2.imshow("output", np.hstack([origImage, image]))
    cv2.waitKey(0)
    

    print(image)    

    L3 = np.array([1, 2, 1]).T
    E3 = np.array([-1, 0, 1]).T
    S3 = np.array([-1, 2, -1]).T

    E5 = np.convolve(E3, L3)
    L5 = np.convolve(L3, L3)
    print(L5)



if __name__ == "__main__":
    filename = "/home/chen2156/laserData/src/laser_values/src/datapoint3/unwarpedImage.png"
    LawsConvo(filename, False)
