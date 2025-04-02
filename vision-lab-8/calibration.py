

import numpy as np
import cv2 as cv
import glob
from shared_libraries import stack_images, standard_show

def get_points(images,board, criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)):
    # print("got here")
    objp = np.zeros((board[0]*board[1],3), np.float32)
    objp[:,:2] = np.mgrid[0:board[0],0:board[1]].T.reshape(-1,2)
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    calibrated_imgs = []
    for fname in images:
        img = cv.imread(fname)
        if img is None:
            print(f"Error: Could not read image {fname}")
            continue  # Skip this image
        gray_big = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        
        gray = cv.resize(gray_big, (600,600))
        # standard_show(gray)
        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, board, None)
    
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)

            corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)
            new_frame = cv.drawChessboardCorners(gray.copy(), (board[0],board[1]), corners2, ret)
            calibrated_imgs.append(new_frame)
            # standard_show(new_frame)
            
    # print(objpoints)
    return objpoints,imgpoints, calibrated_imgs
def calibrate(image, objpoints, imgpoints):
    print("Number of object points:", len(objpoints))
    print("Number of image points:", len(imgpoints))
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, image.shape[::-1], None, None)
    return ret, mtx, dist, rvecs, tvecs
def undistort(frame, mtx, dist):
    h,  w = frame.shape[:2]
    standard_show(frame)
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), alpha=1)
    dst = cv.undistort(frame, mtx, dist, None, newcameramtx)
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    # cv.imwrite('calibresult.png', dst)
    return dst, newcameramtx
def reprojection_error(objpoints,imgpoints, rvecs, tvecs, mtx, dist):
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
        mean_error += error
    total_error = mean_error/len(objpoints)
    # print(total_error)
    # print( "total error: {}".format(mean_error/len(objpoints)) )
    return total_error
def part_1():
    images = glob.glob('*.jpg')
    # images = ["left12.jpg", "left02.jpg"]
    objpoints, imgpoints, frames = get_points(images, board=(7,6))
    image = cv.imread(images[0], cv.IMREAD_GRAYSCALE)
    ret, mtx, dist, rvecs, tvecs= calibrate(image, objpoints, imgpoints)
    undistored, newcameramatrix = undistort(image, mtx, dist)
    error = reprojection_error(objpoints, imgpoints, rvecs, tvecs, mtx, dist)
    compare_image = stack_images([image, undistored], 2)
    # standard_show(compare_image)
    #save dist, newcameramatrix, reprojection error

    np.save("calibrations/dist.npy", dist)
    np.save("calibrations/camera_matrix.npy", newcameramatrix)
    np.save("calibrations/error.npy", error)
    np.savez("B.npz", mtx=newcameramatrix, dist=dist, rvecs=rvecs, tvecs=tvecs)

def part_2():
    images = glob.glob('calibration_data/calibration_data/*.jpg')
    # images = glob.glob('*.jpg')
    print(images[0])

    print(len(images))
    objpoints, imgpoints, frames = get_points(images,board=(11,7))
    # standard_show(frames[0])
    # image = cv.imread(images[0], cv.IMREAD_GRAYSCALE)
    image = cv.imread('IMG_6516.jpg', cv.IMREAD_GRAYSCALE)
    print("Number of object points:", len(objpoints))
    print("Number of image points:", len(imgpoints))
    standard_show(image)
    ret, mtx, dist, rvecs, tvecs= calibrate(image, objpoints, imgpoints)
    undistored, newcameramatrix = undistort(image, mtx, dist)
    error = reprojection_error(objpoints, imgpoints, rvecs, tvecs, mtx, dist)
    compare_image = stack_images([image, undistored], 2)
    standard_show(compare_image)
    #save dist, newcameramatrix, reprojection error
    cv.imwrite("Comparison_image.jpg", compare_image)
    np.save("calibration_data/dist.npy", dist)
    np.save("calibration_data/camera_matrix.npy", newcameramatrix)
    np.save("calibration_data/error.npy", error)
    np.savez("calibration_data/B.npz", mtx=newcameramatrix, dist=dist, rvecs=rvecs, tvecs=tvecs)
def part_3():
    images = glob.glob('my_board/images2/*.jpeg')
    # images = ['my_board/test1.jpeg', 'my_board/test2.jpeg']
    # images = glob.glob('*.jpg')
    print(images[0])

    print(len(images))
    objpoints, imgpoints, frames = get_points(images, board=(7,7))
    # standard_show(frames[0])
    # image = cv.imread(images[0], cv.IMREAD_GRAYSCALE)
    image_big = cv.imread('my_board/images2/test.jpg', cv.IMREAD_GRAYSCALE)
    image = cv.resize(image_big, (600,600))
    print("Number of object points:", len(objpoints))
    print("Number of image points:", len(imgpoints))
    # standard_show(image)
    ret, mtx, dist, rvecs, tvecs= calibrate(image, objpoints, imgpoints)
    undistored, newcameramatrix = undistort(image, mtx, dist)
    error = reprojection_error(objpoints, imgpoints, rvecs, tvecs, mtx, dist)
    compare_image = stack_images([image, undistored], 2)
    standard_show(compare_image)
    #save dist, newcameramatrix, reprojection error
    cv.imwrite("Comparison_image.jpg", compare_image)
    np.save("my_board/dist.npy", dist)
    np.save("my_board/camera_matrix.npy", newcameramatrix)
    np.save("my_board/error.npy", error)
def main():
    part_1()
    # part_2()
    # part_3()


if __name__ == "__main__":
    main()

