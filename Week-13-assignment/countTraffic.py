import numpy as np
import cv2 as cv
import math
from ultralytics import YOLO
LOADED_IMG_SIZE = (960,540)

class stored_ouputs:
    def __init__(self, class_id,class_name, bounding_box):
        
        self.class_id = class_id
        self.class_name = class_name
        self.bounding_box = bounding_box


def load_images(names,channels=cv.IMREAD_COLOR):
    images = []
    for name in names:
        print(name)
        image = cv.imread(name, channels)
        image = cv.resize(image, LOADED_IMG_SIZE) 
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
def load_video_file(file_path):
    cap = cv.VideoCapture(file_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
    
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        gray = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        frame  = cv.resize(frame, LOADED_IMG_SIZE) 
        frames.append(frame)
    cap.release()
    return frames

def play_back_video(frames):
    for frame in frames:
        cv.imshow('frame', frame)
        if cv.waitKey(1) == ord('q'):
            break
    
    cv.destroyAllWindows()
def extract_white_mask(img):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 200])  # tweak if needed
    upper_white = np.array([150, 30, 255])
    mask = cv.inRange(hsv, lower_white, upper_white)
    return mask
# def filter_lines_by_angle(lines, angle_range=(80, 100)):
#     good_lines = []
#     for line in lines:
#         x1, y1, x2, y2 = line[0]
#         angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
#         if angle_range[0] < abs(angle) < angle_range[1]:
#             good_lines.append(line)
#     return good_lines
def lines(frame):
    print(frame.shape)
    white_mask = extract_white_mask(frame)
    # cv.imwrite("masked.png", white_mask)
    # cv.imshow("white", white_mask)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    clean_mask = cv.morphologyEx(white_mask, cv.MORPH_CLOSE, np.ones((5,5),np.uint8))
    edges = cv.Canny(clean_mask, 50, 150)
    lines = cv.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=60, maxLineGap=20)

    output = frame.copy()
    valid_rectangle = np.array([[235,425], [600,300], [900,275],[785,500]])
    rect_cnt = np.array(valid_rectangle, dtype=np.int32).reshape((-1, 1, 2))
    rect = cv.boundingRect(rect_cnt) 
    for line in lines:  # lines from cv2.HoughLinesP or similar
        line = line[0]
        pt1 = tuple(line[0:2])
        pt2 = tuple(line[2:4])
        
        visible, new_pt1, new_pt2 = cv.clipLine(rect, pt1, pt2)

        if visible:
            print("Line is fully inside the rectangle")
            cv.line(output, pt1, pt2, (0, 255, 0), 5)

            
            
    # cv.line(output, (235,425), (600,300), (0,0,0), 5)
    # cv.line(output, (785,500), (900,275), (0,0,255), 5)
    cv.imshow("Detected Crosswalk Lines", output)
    cv.waitKey(0)
    cv.destroyAllWindows()
def load_model():
   model = YOLO("yolo11n.pt")
   return model
def analyze_with_model(model, frame):
    results = model(frame)
    result = results[0] 
    boxes = result.boxes 
    class_ids = boxes.cls.cpu().numpy().astype(int)  
    bboxes = boxes.xyxy.cpu().numpy()              
    names = result.names
    
    items_found = []
    for i, box in enumerate(boxes):
        items_found.append(stored_ouputs(class_id=class_ids[i], class_name=names[class_ids[i]], bounding_box=bboxes[i]))
    for item in items_found:
        print(f"Detected a {item.class_name}  at {item.bounding_box}")
    return items_found
# def output_compact_class(class_id, class_name, bounding_box):
#     item = (class_name, class_id, bounding_box)
#     return item
def draw_bounding_boxes(frame, items):
    for item in items:
        x1,y1, x2,y2 = map(int, item.bounding_box)
        font = cv.FONT_HERSHEY_SIMPLEX
        text = f"{item.class_name}, id: {item.class_id}"
        cv.putText(frame, text, (x1, y1 - 10), font, 0.8, (255, 255, 255), 2, cv.LINE_AA)

        cv.rectangle(frame,(x1,y1),(x2, y2),(0,255,0),3)
    cv.imshow("items", frame)
    cv.waitKey(0)
    cv.destroyAllWindows()
def main():
    # file = "TrafficVideo.mp4"
    # frames = load_video_file(file)
    img = load_images(["single_frame.png"])
    # cv.imwrite("single_frame.png", frames[0])
    # lines(img)
    model = load_model()
    items = analyze_with_model(model, img)
    draw_bounding_boxes(img, items)
    # play_back_video(frames)
    
if __name__ == "__main__":
    main()