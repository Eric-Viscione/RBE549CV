import numpy as np
from shared_libraries import standard_show, load_images, stack_images
import cv2 as cv


def canny_hough( frame, kernel = 3):
# 
    # frame = cv.Canny(frame,1, 1 )
    new_frame = cv.GaussianBlur(frame, (kernel,kernel), 10, 10)
    gray = cv.cvtColor(new_frame, cv.COLOR_BGR2GRAY)
    # standard_show("test",gray)
    edges = cv.Canny(gray, 50, 150, apertureSize=3)

    # This returns an array of r and theta values
    lines = cv.HoughLines(edges, 1, np.pi/180, 200)
    A = []
    b = []
    for r_theta in lines:
        arr = np.array(r_theta[0], dtype=np.float64)
        r, theta = arr
        A.append([np.cos(theta), np.sin(theta)])
        b.append(r)
    A = np.array(A)
    b = np.array(b)
    A_T = np.transpose(A)
    A_T_A_inv = np.linalg.pinv(np.dot(A_T, A))
    t = np.dot(np.dot(A_T_A_inv, A_T), b)
    
    #t= (A_t*A)^-1*A^t*b
    # x,y = int(np.linalg.lstsq(A, b, rcond=None)[0][0]), int(np.linalg.lstsq(A, b, rcond=None)[0][1])
    
    vanishing_point = (int(t[0]), int(t[1]))
    print(vanishing_point)
    cv.circle(frame, vanishing_point, 25, (0,0,255), thickness=-1)
    return frame


        

# def hough():
    
def main():
    names = ["road.jpg", "texas.png"]
    imgs = load_images(names)


    result = []
    for image in imgs:
        frame = canny_hough(image)
        result.append(frame)
    final = stack_images(result, 1)
    standard_show("Vanishing Point", final)
    cv.imwrite("Vanishing_point.jpg", final)

if __name__ == "__main__":
    main()