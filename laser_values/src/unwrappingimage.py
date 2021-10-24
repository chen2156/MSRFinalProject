import numpy as np
import argparse
import cv2
from PIL import Image

def unwrapImage(filePath):
    #detect circle
    image = cv2.imread(filePath)
    output = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # detect circles in the image
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2, 100, minRadius = 200)
    # ensure at least some circles were found
    if circles is not None:
	    # convert the (x, y) coordinates and radius of the circles to integers
	    circles = np.round(circles[0, :]).astype("int")
	    # loop over the (x, y) coordinates and radius of the circles
	    for (x, y, r) in circles:
		    # draw the circle in the output image, then draw a rectangle
		    # corresponding to the center of the circle
		    cv2.circle(output, (x, y), r, (0, 255, 0), 4)
		    cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

	# show the output image
    cv2.imshow("output", np.hstack([image, output]))
    cv2.waitKey(0)

    #Extract circle from image
    i = circles[0]

    height = image.shape[0]
    width = image.shape[1]

    canvas = np.zeros((height, width))

    # Draw the outer circle:
    color = (255, 255, 255)
    thickness = -1
    centerX = i[0]
    centerY = i[1]
    radius = i[2]
    cv2.circle(canvas, (centerX, centerY), radius, color, thickness)

    # Create a copy of the input and mask input:
    imageCopy = image.copy()
    imageCopy[canvas == 0] = (0, 0, 0)

    # Crop the roi:
    x = centerX - radius
    y = centerY - radius
    h = 2 * radius
    w = 2 * radius

    # show the output image
    cv2.imshow("output", np.hstack([image, imageCopy]))
    cv2.waitKey(0)


    #UnWarp the image
    newHeight = round(radius)
    newWidth = int(round(2.0 * (radius / 2) * np.pi))
    print(newWidth)
    print(type(newWidth))
    map_x = np.zeros((newHeight, newWidth), np.float32)
    map_y = np.zeros((newHeight, newWidth), np.float32)


    #Build up the image map
    for y in range(0, int(newHeight - 1)):
        for x in range(0, int(newWidth - 1)):

            r = float(y) / float(newHeight) * radius
            theta = float(x) / float(newWidth) * 2.0 * np.pi

            xS = centerX + r * np.sin(theta)
            yS = centerY + r * np.cos(theta)

            map_x.itemset((y, x), int(xS))
            map_y.itemset((y, x), int(yS))

    #Unwarp

    newUnwarpImage = cv2.remap(imageCopy, map_x, map_y, cv2.INTER_LINEAR)

    #newResult = Image(newUnwarpImage, cv2image=True)


     # show the output image
    cv2.imshow("output", newUnwarpImage)
    cv2.waitKey(0)

    #flip the image upside down

    newUnwarpImage = cv2.flip(newUnwarpImage, 0)

     # show the output image
    cv2.imshow("output", newUnwarpImage)
    cv2.waitKey(0)







    



    


if __name__ == "__main__":
    filename = "/home/chen2156/image.png"
    unwrapImage(filename)