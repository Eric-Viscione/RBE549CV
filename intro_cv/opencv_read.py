import cv2 as cv ##import the cv2 library and reference is as cv
import sys

img = cv.imread(cv.samples.findFile("starry_night.jpg"))#read in the iamge and save the result to img

if img is None:    #check if the image is valid and loaded
    sys.exit("Could not read the image")
cv.imshow("Display window", img) ##show the image and create a loop to kill it when s is pressed
k = cv.waitKey(0)

if k == ord("S"):
    cv.imwrite("Starry_night.jpg", img) 