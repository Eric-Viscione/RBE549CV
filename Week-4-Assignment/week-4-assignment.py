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
    
def check_even(n):
    return n % 2 == 0
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
    ## double the size of the image so there can be more levels to check
    scaled_image = scale_image(gray_img, scale_input  )
    dog_images = []
    prev_img = scaled_image
    for i in range(octave):
        blurred_images = []
        for j in range(level):
            # scale = sigma_init * m.sqrt(2)**(1/level)**((i)*level+j)
            # scale = sigma_init * j**level
            scale = (sigma_init * m.sqrt(2)**(j / level) * (i + 1))*6
            kernel = int(gaussian_constant * scale) | 1  # Ensure odd size
            p = level*(i)
            # kernel = (int(round(scale*gaussian_constant))+1) if check_even(int(round(scale*gaussian_constant))) else int(round(scale*gaussian_constant))
            # kernel = int(6 * scale) + 1
            print(kernel)
            print(scale)
            temp_gaussed = cv.GaussianBlur(scaled_image,(kernel, kernel), scale)
            blurred_images.append(temp_gaussed)
        for k in range(1, len(blurred_images)):
            dogged_temp = cv.subtract(blurred_images[k], blurred_images[k-1])
            dog_images.append(dogged_temp)
            
            # Display the DoG image
            # cv.imshow(f"DoG Level {i}-{k}", dogged_temp)
            # cv.waitKey(0)
            # cv.destroyAllWindows()
        # scaled_image = scale_image(scaled_image, 0.5)
    ##iterate through the DoG pyramid
    dog_images.append(gray_img)
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
