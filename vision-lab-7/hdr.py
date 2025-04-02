import cv2 as cv
import numpy as np
from shared_libraries import load_images

def mertens(images):
    mertens = cv.createMergeMertens()
    res = mertens.process(images)
    res_mertens_8bit = np.clip(res*255, 0, 255).astype('uint8')
    cv.imwrite("hdr_mertesens.jpg", res_mertens_8bit)
    return res_mertens_8bit

def main():
    images = load_images(['IMAGE_1.JPG', 'IMAGE_2.JPG', 'IMAGE_3.JPG'])
    merten = mertens(images)
    cv.imshow("Mertens", merten)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()