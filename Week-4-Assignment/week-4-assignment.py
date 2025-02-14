import numpy as np
import cv2 as cv
import os
import math 
import argparse
from scipy.signal import convolve2d
class constants:
    octave = 3
    sigma_init = math.sqrt(2) 
    level = 3
    gaussian_constant = 12
    row = 256
    column = 256
    contrast_threshold=0.04
    orig_image = None
    r = 10
    accuracy_threshold = 0.01

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

def scale_space_extrema(img, constants):
    ##constants
    octave = constants.octave
    
    sigma_init = constants.sigma_init
    level = constants.level
    gaussian_constant = constants.gaussian_constant
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
            temp_gaussed = cv.GaussianBlur(curr_img,(kernel, kernel), scale)
            blurred_images.append(temp_gaussed)
        ##iterate through the DoG pyramid
        for k in range(1, len(blurred_images)):
            dogged_temp = cv.subtract(blurred_images[k], blurred_images[k-1])
            dog_images.append(dogged_temp)
        curr_img_scaled = cv.resize(curr_img, None, fx=0.5, fy=0.5, interpolation=cv.INTER_LINEAR)
    print(type(dog_images))
    return dog_images

# def find_extrema():
def key_point_localization(dogged_images, constant):
    row = constant.row  # 256
    column = constant.column  # 256
    octave = constant.octave  # 3
    interval = constant.level - 1  # 3 - 1
    keypoint_store = 4
    level = constant.level

    number = 0
    for i in range(2, octave + 2):  # i = 2 to octave (inclusive in MATLAB)
        number += (2 ** (i - octave) * column) * (2 * row) * interval

    extrema = np.zeros(int(number)*4)  #creates vector to store all the extrma found
    flag = 0
    for idx in range(octave):
        m, n  = dogged_images[idx].shape
        m=m-2
        n=n-2
        # volume=int(m*n/(4**(i-1)))
        volume = int(m * n / (keypoint_store** (idx - 1)))
        for k in range(2, interval+1):
            for j in range(volume):
                x = math.ceil((j+1)/n)
                y = ((j - 1) % m) + 1  # Equivalent to MATLAB's mod(j-1, m) + 1
                # print(x, y)
                # sub = dogged_images[i][x:x+3, y:y+3, k-1:k+2]  
                # sub = dogged_images[i][x:x+2, y:y+2] 
                ##get the maximum value of the 
                
                sub_img = dogged_images[idx][x:x+2, y:y+2]
                    
                if sub_img.size > 0:
                    center_value = dogged_images[idx][x, y]
                    max_value = np.max(sub_img.size)
                    min_value = np.min(sub_img.size)
                    
                    # Check if current pixel is local maximum
                    if center_value == max_value:
                        extrema[flag:flag+keypoint_store] = [idx, k, j, 1]
                        flag += keypoint_store
                        
                    # Check if current pixel is local minimum
                    if center_value == min_value:
                        extrema[flag:flag+keypoint_store] = [idx, k, j, -1]
                        flag += keypoint_store
    accurate_keypoints = accurate_keypoint_filtering(dogged_images, extrema, constant)
def accurate_keypoint_filtering(dogged_images, extrema, constant):

    """Seperated out this function because it does not work well at all

    Returns:
        _type_: _description_
    """
    m, n, octave, level = constant.row, constant.column, constant.octave, constant.level ##get the constants for this run
    r_threshold = (constant.r + 1)**2 / constant.r
     
    valid_mask = extrema[3::4] != 0  ##filter out all the zero values so we dont overflow anywhere
    extrema = extrema[np.repeat(valid_mask, 4)]

    extr_volume = len(extrema) // 4

    for i in range(extr_volume):
        #my input is 2d idk why
        base_idx = 4 * i
        octave_idx = int(extrema[base_idx]) - 1
        # rz = int(extrema[base_idx + 1])
        pos = int(extrema[base_idx + 2]) - 1
        x = pos // (n // (2 ** (extrema[base_idx] - 2)))
        y = pos % (m // (2 ** (extrema[base_idx] - 2)))
        rx = int(x)
        ry = int(y)  
        ##I did something wrong somewhere becasue I only have 2d array :?
        if (0 <= octave_idx < len(dogged_images)):
            D = dogged_images[octave_idx]

            if (0 <= rx - 1 and rx + 1 < D.shape[0] and 0 <= ry - 1 and ry + 1 < D.shape[1] ):
                ##calcuilate the next derivatives
                Dxx = D[rx - 1, ry] + D[rx + 1, ry] - 2 * D[rx, ry]
                Dyy = D[rx, ry - 1] + D[rx, ry + 1] - 2 * D[rx, ry]
                Dxy = (D[rx - 1, ry - 1] + D[rx + 1, ry + 1] - D[rx - 1, ry + 1] - D[rx + 1, ry - 1])
                ##calculat the determinate
                deter = Dxx * Dyy - Dxy * Dxy
                R = (Dxx + Dyy) / deter if deter != 0 else float('inf')

                if deter < 0 or R > r_threshold:
                    extrema[base_idx + 3] = 0

   
    valid_mask = extrema[3::4] != 0  ##remove the zeros again
    extrema = extrema[np.repeat(valid_mask, 4)]
    ##calculcate the final coords and then ensure that they exist before displaying them
    x_coords = np.floor((extrema[2::4] - 1) / (n / (2 ** (extrema[0::4] - 2))))
    y_coords = np.mod((extrema[2::4] - 1), m / (2 ** (extrema[0::4] - 2)))

    rx = x_coords / (2 ** (octave - 1 - extrema[0::4]))
    ry = y_coords / (2 ** (octave - 1 - extrema[0::4]))

    keypoints = [cv.KeyPoint(float(rx[i]), float(ry[i]), 1) for i in range(len(rx))]
    
    if len(keypoints) > 0:
        print("keypointing")
        image_with_keypoints = cv.drawKeypoints(constant.orig_image, keypoints, None, flags=cv.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
        cv.imshow("Keypoints", image_with_keypoints)
        cv.waitKey(0)
        cv.destroyAllWindows()

    return extrema, rx, ry



def main():
    parser = argparse.ArgumentParser(description ='Scale Space Extrema Detection and keypoint localization')
    parser.add_argument('--image', type=str , default="Fabio.png",  required=False ,help='Path to the desired image' )
    # path = "Fabio.png"

    args = parser.parse_args()
    constants_init = constants()
    path = args.image
    print(os.getcwd())
    img = cv.imread(path)
    assert img is not None, "file could not be read, check with os.path.exists()"
    resized_img = cv.resize(img, (constants_init.row, constants_init.column))
    constants_init.orig_image = resized_img
    ret = scale_space_extrema(resized_img, constants_init)
    extrema = key_point_localization(ret, constants_init)
    stacked_image = stack_images(ret, 3)
    cv.imshow("DOGGED", stacked_image)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
