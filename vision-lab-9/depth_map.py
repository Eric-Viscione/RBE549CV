import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from shared_libraries import load_images
 

imgL, imgR = load_images(['aloeL.jpg', 'aloeR.jpg'], cv.IMREAD_GRAYSCALE)
stereo = cv.StereoBM.create(numDisparities=16, blockSize=15)
disparity = stereo.compute(imgL,imgR)
fig, _ = plt.subplots()
plt.imshow(disparity,'gray')
plt.show()
fig.savefig("figure.png", dpi=300) 