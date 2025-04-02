import cv2 as cv
import numpy as np
coins = ["descriptors_and_coins/penny_front","descriptors_and_coins/nickel_front","descriptors_and_coins/quarter_front"]
for coin in coins:
    image = cv.imread(f"{coin}.jpg", cv.IMREAD_GRAYSCALE)

    # Initialize SIFT
    sift = cv.SIFT_create()

    # Detect keypoints and compute descriptors
    keypoints, descriptors = sift.detectAndCompute(image, None)

    # Save descriptors to a file
    np.save(f"{coin}_descriptors.npy", descriptors)