import cv2 as cv
import numpy as np


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

def get_corners(h,w):
    return np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
def get_homography( img1, img2, draw_matches = False):
    MIN_MATCH_COUNT = 10
    gray_img1 = cv.cvtColor(img1.copy(),cv.COLOR_BGR2GRAY)
    gray_img2 = cv.cvtColor(img2.copy(),cv.COLOR_BGR2GRAY)

    # Initiate SIFT detector
    sift = cv.SIFT_create()
    
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(gray_img1,None)
    kp2, des2 = sift.detectAndCompute(gray_img2,None)
    
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    
    flann = cv.FlannBasedMatcher(index_params, search_params)
    
    matches = flann.knnMatch(des1,des2,k=2)
    
    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)
    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
    
        h,w = gray_img1.shape
        pts = get_corners(h,w)
        dst = cv.perspectiveTransform(pts,M)
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        corners = get_corners(h1, w1)
        warped_corners = cv.perspectiveTransform(corners, M)
        x_min, y_min = np.int32(warped_corners.min(axis=0).ravel())
        x_max, y_max = np.int32(warped_corners.max(axis=0).ravel())
        canvas_width = max(x_max, w2) - min(x_min, 0)
        canvas_height = max(y_max, h2) - min(y_min, 0)
        offset_x = -x_min
        offset_y = -y_min
        translation_matrix = np.array([[1, 0, offset_x], [0, 1, offset_y], [0, 0, 1]])
        M_translated = translation_matrix @ M  
        img1_warped = cv.warpPerspective(img1, M_translated, (canvas_width, canvas_height), borderMode=cv.BORDER_CONSTANT, borderValue=(0, 0, 0))
        stitched_canvas = img1_warped.copy()
        stitched_canvas[offset_y:offset_y + h2, offset_x:offset_x + w2] = img2
        img3 = cv.warpPerspective(img2, M, (img1.shape[1], img1.shape[0]))
       
        return stitched_canvas
    
    else:
        print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
        matchesMask = None
        return


def main():
    
    img1 = cv.imread('boston1.jpeg')
    img2 = cv.imread('boston2.jpeg')
    assert img1 is not None and img2 is not None, "file could not be read, check with os.path.exists()"
    panorama_img = get_homography(img1, img2)
    cv.imshow("Panorama", panorama_img)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()