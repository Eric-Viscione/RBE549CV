import numpy as np
import cv2 as cv
import os
import math as m

def stack_images(images, cols):
    rows = (len(images) + cols - 1) // cols  
    size = (300, 300) 
    imgs_resized = [cv.resize(img, size) for img in images]
    empty = np.zeros_like(imgs_resized[0])
    imgs_resized += [empty] * (rows * cols - len(images))  
    spacing_line = np.zeros((size[1], 25), dtype=np.uint8) 
    
    stacked_images = []
    for i in range(rows):
        row = np.hstack(imgs_resized[i * cols:(i + 1) * cols]) 
        stacked_images.append(row)
    return np.vstack(stacked_images)

def scale_image(img, scale):
    img_copy = img.copy()
    if len(img.shape) == 3:
        img_height, img_width, _ = img_copy.shape
    else:
        img_height, img_width = img_copy.shape
    scaled_size = (img_height*scale, img_width*scale)
    doubled_img = cv.resize(img_copy, scaled_size, interpolation=cv.INTER_LINEAR)
    return doubled_img

def scale_space_extrema(img):
    ##constants
    scale_input = 2
    octave = 3
    sigma_init = m.sqrt(2) 
    level = 3
    gaussian_constant = 6
    ##turn the image gray for processing
    gray_img = cv.cvtColor(img.copy(),cv.COLOR_BGR2GRAY)
    curr_img_scaled = cv.resize(gray_img, None, fx=2, fy=2, interpolation=cv.INTER_LINEAR)
    ## double the size of the image so there can be more levels to check
    dog_images = []
    for i in range(octave):
        blurred_images = []
        curr_img = curr_img_scaled.copy()
        for j in range(level+3):
            scale = sigma_init  * (2 ** (j / level))
            kernel = int(gaussian_constant * scale) | 1  # Ensure odd size


            # print(f"kernel: {kernel}")
            # print(f"Scale: {scale}")
            temp_gaussed = cv.GaussianBlur(curr_img,(kernel, kernel), scale)
            blurred_images.append(temp_gaussed)
        ##iterate through the DoG pyramid
        for k in range(1, len(blurred_images)):
            dogged_temp = cv.subtract(blurred_images[k], blurred_images[k-1])
            dog_images.append(dogged_temp)
            
            # Display the DoG image
            # cv.imshow(f"DoG Level {i}-{k}", dogged_temp)
            # cv.waitKey(0)
            # cv.destroyAllWindows()
        # scaled_image = scale_image(scaled_image, 0.5)
        curr_img_scaled = cv.resize(curr_img, None, fx=0.5, fy=0.5, interpolation=cv.INTER_LINEAR)
    # dog_images.append(gray_img)
    return dog_images


def main():
    path = 'Fabio.png'
    img = cv.imread(path)
    assert img is not None, "file could not be read, check with os.path.exists()"
    ret = scale_space_extrema(img)
    stacked_image = stack_images(ret, 3)
    
    cv.imshow("DOGGED", stacked_image)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
