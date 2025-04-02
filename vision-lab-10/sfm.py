import numpy as np
import cv2 as cv

import open3d as o3d

def load_images(names):
    images = []
    for name in names:
        print(name)
        image_dist = cv.imread(name)
        if image_dist is None:
            print(f'Could Not load image {name}. Check the path ')
            continue
        _,image = undistort_images(image_dist)

        images.append(image)
    if not images:
        raise ValueError("No valid images were loaded. Please check the file paths.")
    return images
def find_keypoints(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    sift = cv.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return keypoints, descriptors
    # np.save(f"{coin}_descriptors.npy", descriptors)
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
def feature_matching(images, method_match='flann', method_extract='sift', save = True):
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
        keypoints = [kp_0, kp_1]
        return keypoints, matches,img3
def undistort_images(img):
    h, w = img.shape[:2]
    with np.load('B.npz') as X:
        mtx, dist, _, _ = [X[i] for i in ('mtx','dist','rvecs','tvecs')]
    new_camera_mtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    undistorted_img = cv.undistort(img, mtx, dist, None, new_camera_mtx)
    return new_camera_mtx, undistorted_img

def find_fundamental_matrix(matches, keypoints):
    good_matches = []
    for m in matches:
        if isinstance(m, (list, tuple)) and len(m) > 0:
            good_matches.append(m[0])
        else:
            good_matches.append(m)
    kp0, kp1 = keypoints
    pts1 = np.float32([kp0[m.queryIdx].pt for m in good_matches])
    pts2 = np.float32([kp1[m.trainIdx].pt for m in good_matches])

    F, mask = cv.findFundamentalMat(pts1, pts2, cv.FM_LMEDS)  

    pts1 = pts1[mask.ravel() == 1]
    pts2 = pts2[mask.ravel() == 1]

    return F, pts1, pts2

def verify_epipolar_constraint(F, pts1, pts2):
    ones = np.ones((pts1.shape[0], 1))
    
    pts1_h = np.hstack([pts1, ones])  # (N,3)
    pts2_h = np.hstack([pts2, ones])  # (N,3)

    # Compute q_r^T * F * q_l
    errors = np.abs(np.sum(pts2_h @ F * pts1_h, axis=1))

    return errors  
def compute_essential_matrix(F, K):
    E = K.T @ F @ K
    
    det_E = np.linalg.det(E)
    print("Determinant of E:", det_E)
    
    return E
def recover_pose(E, pts1, pts2, K):

    points, R, t, mask = cv.recoverPose(E, pts1, pts2, K)
    print(" Rotation:\n", R)
    print(" Translation:\n", t)
    return R, t, mask
def find_projection_matricies(mtx, R1, R2, t):
        # Camera 0: P0 = K [I | 0]
    P0 = mtx @ np.hstack([np.eye(3), np.zeros((3, 1))])
    
    P1_candidates = []
    for R_candidate in [R1, R2]:
        for sign in [1, -1]:
            t_candidate = sign * t
            P1 = mtx @ np.hstack([R_candidate, t_candidate])
            P1_candidates.append(P1)
    
    return P0, P1_candidates
def LinearLSTriangulation(P0, P1, pt1, pt2):
        # Extract image coordinates
    u1, v1 = pt1
    u2, v2 = pt2

    row1 = u1 * P0[2, :] - P0[0, :]
    row2 = v1 * P0[2, :] - P0[1, :]
    row3 = u2 * P1[2, :] - P1[0, :]
    row4 = v2 * P1[2, :] - P1[1, :]

    A = np.vstack([row1, row2, row3, row4])

    U, S, Vt = np.linalg.svd(A)
    X = Vt[-1]  # 4-element homogeneous coordinate
    return X

def find_points(P0, P1, pts1, pts2):
    points_3d = []
    for pt1, pt2 in zip(pts1, pts2):
        X = LinearLSTriangulation(P0, P1, pt1, pt2)
        points_3d.append(X)
    points_3d = np.array(points_3d)
    np.array(points_3d)
def check_front(P, X, K):
    X_euclid = X[:3] / X[3]
    X = X / X[3] 
    P0_canonical = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    if np.allclose(P, P0_canonical, atol=1e-6):
        return X_euclid[2] > 0
    else:
        Rt = np.linalg.inv(K) @ P  # This yields [R | t]
        R = Rt[:, :3]
        t = Rt[:, 3]
        X_cam = R @ X_euclid + t
        return X_cam[2] > 0
def find_front_p1(P0, candidate_P1s, pts1, pts2, K):
    best_count = -1
    best_P1 = None
    best_points_3d = None

    # Loop over each candidate P1
    for candidate_P1 in candidate_P1s:
        points_3d = []  
        count_in_front = 0
        
        for pt1, pt2 in zip(pts1, pts2):
            X = LinearLSTriangulation(P0, candidate_P1, pt1, pt2)
            X_hom = np.hstack([X, [1]])
            if check_front(P0, X, K) and check_front(candidate_P1, X, K):
                count_in_front += 1
            points_3d.append(X)
        
        print("Candidate P1 has", count_in_front, "points in front of both cameras out of", len(pts1))
        
        if count_in_front > best_count:
            best_count = count_in_front
            best_P1 = candidate_P1
            best_points_3d = np.array(points_3d)
    
    return best_P1, best_points_3d
def find_reprojection_error(P, pts_2d, pts_3d):
    total_error = 0
    num_points = len(pts_2d)
    
    for pt_2d, X in zip(pts_2d, pts_3d):
        if len(X) == 4:
            X_hom = X
        else:
            X_hom = np.hstack([X, [1]])
        
        projected = P @ X_hom  
        projected /= projected[2] 
        error = np.linalg.norm(pt_2d - projected[:2])
        total_error += error
    return total_error / num_points 
def save_pcd_file(filename, points_3d, colors):
    with open(filename, "w") as file:
        # PCD Header
        file.write("# .PCD v0.7 - Point Cloud Data file format\n")
        file.write("VERSION 0.7\n")
        file.write("FIELDS x y z rgb\n")
        file.write("SIZE 4 4 4 4\n")
        file.write("TYPE F F F U\n")
        file.write("COUNT 1 1 1 1\n")
        file.write(f"WIDTH {len(points_3d)}\n")
        file.write("HEIGHT 1\n")
        file.write("VIEWPOINT 0 0 0 1 0 0 0\n")
        file.write(f"POINTS {len(points_3d)}\n")
        file.write("DATA ascii\n")

        for point, color in zip(points_3d, colors):
            r, g, b = color
            rgb = (r << 16) | (g << 8) | b 
            file.write(f"{point[0]} {point[1]} {point[2]} {rgb}\n")
def find_colors(img, pts):
    colors = []
    for pt in pts:
        x, y = int(pt[0]), int(pt[1])  
        if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:  
            color = img[y, x]  # Get color (B, G, R) from OpenCV (BGR format)
            color = (int(color[2]), int(color[1]), int(color[0]))  
            colors.append(color)
        else:
            colors.append((0, 0, 0)) 
    return colors

def visualize_pcd(file_path, view="front"):

    pcd = o3d.io.read_point_cloud(file_path)

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)

    ctr = vis.get_view_control()
    if view == "front":
        ctr.set_front([0, 0, -1]) 
        ctr.set_up([0, -1, 0])      # Y-axis as up
    elif view == "top":
        ctr.set_front([0, -1, 0])  
        ctr.set_up([0, 0, -1])     # Z-axis as up
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(f"{view}.png")
    vis.run()
    vis.destroy_window()
def visualize_3d(file_path):
    pcd = o3d.io.read_point_cloud(file_path)
    o3d.visualization.draw_geometries([pcd])
def main():
    with np.load('B.npz') as X:
        mtx, dist, _, _ = [X[i] for i in ('mtx','dist','rvecs','tvecs')]
    images = load_images(["IMG_0552.jpg", "IMG_0553.jpg"])

    keypoints,matches, img = feature_matching(images)
    cv.imwrite("keypoints_and_matches.jpg", img)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    F, pts1, pts2 = find_fundamental_matrix(keypoints=keypoints, matches=matches)
    print(f"Fundemental Matrix {F}")
    error = verify_epipolar_constraint(F, pts1=pts1, pts2=pts2)
    print("Errors", error)
    E = compute_essential_matrix(F,mtx )
    print(f"Essential Matrix {E}")
    R,t,mask = recover_pose(E, pts1, pts2, mtx)
    R2 = np.transpose(R)
    P0, p1_candidates = find_projection_matricies(mtx, R, R2, t)
    print(f"P0 = {P0} \n, p1s = {p1_candidates}")
    # points = find_points(P0, p1_candidates[0],e pts1, pts2)
    p1_front, front_points = find_front_p1(P0, p1_candidates, pts1, pts2, mtx)
    p1_error = find_reprojection_error(p1_front, pts2, front_points)
    p0_error = find_reprojection_error(P0, pts1, front_points)
    print(f"P0 Reprojection Error:\n {p0_error} \n P1 Reprojection Error:\n {p1_error}")
    colors = find_colors(images[0], pts1)
    save_pcd_file("front_points.pcd", front_points, colors)
    file = "front_points.pcd"
    visualize_3d(file)
    # visualize_pcd(file, view="front")
    # visualize_pcd(file, view="top")


if __name__ == "__main__":
    main()

