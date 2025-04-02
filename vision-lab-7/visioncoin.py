import sys
import cv2 as cv
import numpy as np
import time
def find_key_by_value(d, target_value):
    for key, values in d.items():
        if target_value in values:
            return key  # Return the matching key
    return None  # Not found

def crop_coin(frame, circle):
    buffer = 20
    x_center, y_center = circle[0], circle[1]
    radius = circle[2]+buffer

    # Compute bounding box
    x_start = max(0, x_center - radius)
    y_start = max(0, y_center - radius)
    x_end = min(frame.shape[1], x_center + radius)
    y_end = min(frame.shape[0], y_center + radius)

    # Crop the image using array slicing
    cropped_img = frame[y_start:y_end, x_start:x_end]
    return cropped_img
def flann(img):
    descrip_path = "descriptors_and_coins"
    coins_dict_full ={
    'penny_front': 1,
    'penny_back': 1,
    'nickel_front': 5,
    'nickel_back': 5,
    'dime_front': 10,
    'dime_back': 10,
    'quarter_front': 25,
    'quarter_back': 25
    }
    coins_dict = {
        'penny_front': 1,
        'nickel_front': 5,
        'quarter_front': 25
    }
    coins_long = ["penny_front", "penny_back", "nickel_front", "nickel_back", "dime_front", "dime_back", "quarter_front", "quarter_back"]
    coins = ["penny_front", "nickel_front", "quarter_front"]
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    new_descriptors = find_keypoints(img)
    if new_descriptors is None or new_descriptors.shape[0] < 2:
        print("Not enough descriptors detected in the image.")
        return None
    best_match = 0
    coin_match = None
    for coin in coins:
        stored_descriptors = np.load(f"{descrip_path}/{coin}_descriptors.npy")
        matches = flann.knnMatch(stored_descriptors, new_descriptors, k=2)

        # Filter matches using Lowe’s ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
        if len(good_matches) > best_match:
            best_match = len(good_matches)
            coin_type = coin
            
            coin_match = coins_dict[coin_type]
    print("At detection" , coin_type)
    return coin_match
def find_keypoints(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    sift = cv.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return descriptors
    # np.save(f"{coin}_descriptors.npy", descriptors)



def hough_circle(frame):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray = cv.medianBlur(gray, 5)
    rows = gray.shape[0]
    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, rows / 8,
                               param1=100, param2=45,
                               minRadius=1, maxRadius=200)
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            # circle center
            cv.circle(frame, center, 1, (0, 100, 100), 3)
            # circle outline
            radius = i[2]
            cv.circle(frame, center, radius, (255, 0, 255), 3)

    return circles, frame
def end(cap):
    """Destroys any frames remaining and frees them
    """
    cap.release()
    cv.destroyAllWindows()
def main(argv):

    cap = cv.VideoCapture(0)
    cv.namedWindow('EV_Capture')
    target_fps = 3
    delay = 1 / target_fps

    while True:
        start_time = time.time()
        ret , frame = cap.read()
        if not ret:
            print("Can't recieve frame (stream end?). Exiting .....")
            break
        circles, frame = hough_circle(frame)
        coin_value = None
        total = 0
        # cap.set(cv.CAP_PROP_EXPOSURE, -4)
        if circles is not None and len(circles) > 0:
            for coin in circles[0, :]:
                search_frame = frame.copy()
                cropped_coin = crop_coin(search_frame, coin)
                if cropped_coin is not None:
                    coin_value = flann(cropped_coin)
                    if coin_value is not None:
                        total += coin_value
        else:
            print("No coins Found")
            
        print(total)
        if frame is None:
            print("Error: Frame is empty. Check your video source.")
            return
        
        cv.putText(frame, str(total), (0,50),cv.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
        cv.imshow('frame', frame)
        processing_time = time.time() - start_time
        sleep_time = max(0, delay - processing_time)

        time.sleep(sleep_time)
        time.sleep(1/3)
        if cv.waitKey(1) == ord('q'):
            break

    end(cap)


    
    return 0
if __name__ == "__main__":
    main(sys.argv[1:])