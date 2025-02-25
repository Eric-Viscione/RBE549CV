import numpy as np
import cv2 as cv
 

def stack_images(images, cols):
    rows = (len(images) + cols - 1) // cols  
    # size = (300, 300) turn this on to resize to an arbitrary size
    scale_factor = 0.5
    size = (int(images[0].shape[1]*scale_factor), int(images[0].shape[0]*scale_factor))  # (width, height)
    imgs_resized = [cv.resize(img, size) for img in images]
    empty = np.zeros_like(imgs_resized[0])
    imgs_resized += [empty] * (rows * cols - len(images))  
    spacing_line = np.zeros((size[1], 25), dtype=np.uint8) 

    stacked_images = []
    for i in range(rows):
        row = np.hstack(imgs_resized[i * cols:(i + 1) * cols]) 
        stacked_images.append(row)
    return np.vstack(stacked_images)


def feature_extracting(img, method, save_image=False, optimize_surf_threshold = False, optimize_threshold = 50):
    """Takes in an image, and returns an image with keypoints marked and a list of keypoints and a list of descriptors

    Args:
        img (_type_): _description_
        method (_type_): sift or surf
    """
    gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)

    if method == 'sift':
        sift = cv.SIFT_create()
        kp = sift.detect(gray,None)
        img=cv.drawKeypoints(gray,kp,img)
        if save_image:
            cv.imwrite('sift_keypoints.jpg',img)
        kp, des = sift.detectAndCompute(gray,None)
        return img, kp, des
    elif method == 'surf':
        hessian_threshold = 400
        surf = cv.xfeatures2d.SURF_create(hessian_threshold)
        surf.setExtended(True)
        # surf.setUpright(True)
        kp, des = surf.detectAndCompute(img,None)
        # print(len(kp))
        num_iterations = 0
        while len(kp) > optimize_threshold and optimize_surf_threshold:
            num_iterations += 1
            hessian_threshold *= 5/num_iterations
            surf.setHessianThreshold(hessian_threshold)
            kp, des = surf.detectAndCompute(img,None)
            # print(len(kp))
        img=cv.drawKeypoints(gray,kp,img)
        if save_image:
            cv.imwrite('sift_keypoints.jpg',img)
        kp, des = surf.detectAndCompute(gray,None)
        # print( surf.descriptorSize() )
        return img, kp, des
    else:
        print("You need to enter a correct type for mathing")
        raise 
def feature_matching(images, method_match, method_extract, save = True):
    img0 = images[0]
    img1 = images[1]
    _, kp_0, desc_0 = feature_extracting(img0, method_extract)
    _, kp_1, desc_1 = feature_extracting(img1, method_extract)
    if method_match == 'brute_force':
        bf = cv.BFMatcher(cv.NORM_L2, crossCheck=False)
        matches = bf.knnMatch(desc_0,desc_1,k=2)
        draw_params = ratio_test(matches)
        img3 = cv.drawMatchesKnn(img0, kp_0, img1, kp_1, matches, None, **draw_params)
        return img3
    elif method_match == 'flann':
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50) #can pass empty
        # search_params = dict()
        flann = cv.FlannBasedMatcher(index_params,search_params)
        matches = flann.knnMatch(desc_0,desc_1,k=2)
        draw_params = ratio_test(matches)
        img3 = cv.drawMatchesKnn(img0, kp_0, img1, kp_1, matches, None, **draw_params)
        return img3
    


def ratio_test(matches, match_threshold=0.75):
    matchesMask = [[0, 0] for _ in range(len(matches))] 
    for i, (m, n) in enumerate(matches):
        if m.distance < match_threshold * n.distance: 
            matchesMask[i] = [1, 0]  
    draw_params = dict(
        matchColor=(0, 255, 0),  # Green for good matches
        singlePointColor=(255, 0, 0),  # Red for keypoints
        matchesMask=matchesMask,  # Use the match mask to highlight good matches
        flags=cv.DrawMatchesFlags_DEFAULT
    )
    
    return draw_params



def main():
    save = True
    img_0 = cv.imread('book-1.jpg')
    assert img_0 is not None, "file could not be read, check with os.path.exists()"
    img_1 = cv.imread('table-2.jpg')
    assert img_1 is not None, "file could not be read, check with os.path.exists()"
    # img_sift , key_points_sift, descriptors_sift = feature_extracting(img, 'sift')
    # img_surf, key_points_surf, descriptors_surf = feature_extracting(img, 'surf')
    # img = stack_images([img_sift, img_surf], 1)
    # # print(descriptors)
    # cv.imshow("Sift and Surf feature extraction", img)
    match_methods = ['brute_force', 'flann']
    extract_methods = ['sift', 'surf']

    images = [img_0, img_1]
    matched_image = feature_matching(images, 'flann', 'surf')
    for i in range(len(match_methods)):
        for j in range(len(extract_methods)):
            matched_image = feature_matching(images, match_methods[i], extract_methods[j])
            title = f"{extract_methods[j]} and {match_methods[i]}.jpg"
            cv.imwrite(title, matched_image) 
            cv.imshow(title, matched_image)
            cv.waitKey(0)
            cv.destroyAllWindows()
    
    

if __name__ == "__main__":
    main()