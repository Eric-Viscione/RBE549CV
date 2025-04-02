import numpy as np
import cv2 as cv
 

def stack_images(images, cols, scale_factor = .75):
    rows = (len(images) + cols - 1) // cols  
    # size = (300, 300) turn this on to resize to an arbitrary size
    # scale_factor = 0.5
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


def load_images(names):
    images = []
    for name in names:
        print(name)
        image = cv.imread(name)
        if image is None:
            print(f'Could Not load image {name}. Check the path ')
            continue
        images.append(image)
    if not images:
        raise ValueError("No valid images were loaded. Please check the file paths.")
    if len(images) <= 1:
        image = images[0]
        return image
    return images
def standard_show(image, name="image"):

    cv.imshow(name, image)
    cv.waitKey(0)
    cv.destroyAllWindows()
def apply_kernel( frame, kernel, padding = True, fast_algo = True):
    """A generic function to apply a generic kernel to any generic image

    Args:
        frame (_type_): The input frame
        kernel (_type_): Any size kernel to be used
        padding (bool, optional): Add padding to the image to ensure it stays the same size
        fast_algo (bool, optional): This is a faster implementation of kernel actions, much faster than the default one. Defaults to True.

    Returns:
        _type_: _description_
    """
    height, width = frame.shape[0], frame.shape[1]
    kernel_height, kernel_width = kernel.shape[0], kernel.shape[1]
    if fast_algo:   ##this implementation is inspired by https://stackoverflow.com/questions/64587303/python-image-convolution-using-numpy-only
        result_height, result_width = height - kernel_height + 1, width - kernel_width + 1
        ix0 = np.arange(kernel_height)[:, None] + np.arange(result_height)[None, :]
        ix1 = np.arange(kernel_width)[:, None] + np.arange(result_width)[None, :]
        res = kernel[:, None, :, None] * frame[(ix0.ravel()[:, None], ix1.ravel()[None, :])].reshape(kernel_height, result_height, kernel_width, result_width)
        new_img = res.transpose(1, 3, 0, 2).reshape(result_height, result_width, -1).sum(axis = -1)
    else:      #this implementation is custom but incredibly slow limits fps to ~1
        new_img = np.zeros((height, width))
        if padding:
            padded_frame = np.pad(frame,((kernel_height//2, kernel_height//2), (kernel_width//2, kernel_width//2)), mode = 'constant', constant_values=0 )
        else:
            padded_frame = frame
        for i in range(kernel_height//2, height-kernel_height//2):
            for j in range(kernel_width//2, width-kernel_width//2):
                kernel_area = padded_frame[i - kernel_height//2 : i+kernel_height//2+1 ,j-kernel_width//2 : j+kernel_width//2+1]  
                new_img[i,j] = np.clip(int((kernel_area * kernel).sum()), 0, 255) #very slow manual implementation

    new_img = new_img.astype(np.uint8)
    return new_img