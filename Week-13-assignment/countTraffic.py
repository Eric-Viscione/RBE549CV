import numpy as np
import cv2 as cv
import math
from ultralytics import YOLO
from tracker import Tracker
from collections import Counter
from random import randint
from tqdm import tqdm
LOADED_IMG_SIZE = (960,540)
class_entry_counts = {}
seen_ids_by_class = {}
VALID_CLASSES = ['car', 'bicycle', 'person']
people_dimensions = []
car_dimensions = []
class stored_ouputs:
    def __init__(self, class_id,class_name, bounding_box, uuid):
        
        self.class_id = class_id
        self.class_name = class_name
        self.bounding_box = bounding_box
        self.uuid = uuid
        self.tracked_before = False

def write_video(frames, output_path, fps=30, size=(960, 540)):
    fourcc = cv.VideoWriter_fourcc(*'mp4v')  # or 'XVID' for .avi
    out = cv.VideoWriter(output_path, fourcc, fps, size)

    for i, frame in enumerate(frames):
        if frame is None:
            print(f"Warning: Skipping empty frame at index {i}")
            continue
        # resized = cv.resize(frame, size)
        out.write(frame)

    out.release()
    print(f"Video saved to {output_path}")
def playback_video(path):
    cap = cv.VideoCapture(path)

    if (cap.isOpened()== False):
        print("Error opening video file")

    while(cap.isOpened()):
        
        ret, frame = cap.read()
        if ret == True:
            cv.imshow('Frame', frame)
            
            if cv.waitKey(25) & 0xFF == ord('q'):
                break

        else:
            break

    cap.release()

    # Closes all the frames
    cv.destroyAllWindows()
def load_images(names,channels=cv.IMREAD_COLOR):
    images = []
    for name in names:
        # print(name)
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
        # gray = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
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

    output = frame
    valid_rectangle = np.array([[235,425], [600,300], [900,275],[785,500]])
    rect_cnt = np.array(valid_rectangle, dtype=np.int32).reshape((-1, 1, 2))
    rect = cv.boundingRect(rect_cnt) 
    good_lines = []
    for line in lines:  
        line = line[0]
        pt1 = tuple(line[0:2])
        pt2 = tuple(line[2:4])
        
        visible, new_pt1, new_pt2 = cv.clipLine(rect, pt1, pt2)

        if visible:
            # print("Line is fully inside the rectangle")
            good_lines.append((pt1, pt2))
            # cv.line(output, pt1, pt2, (0, 255, 0), 5)

            
            
    # cv.line(output, (235,425), (600,300), (0,0,0), 5)
    # cv.line(output, (785,500), (900,275), (0,0,255), 5)
    # cv.imshow("Detected Crosswalk Lines", output)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    return good_lines
def draw_lines(frame, good_lines):
    for line in good_lines:
        pt1, pt2 = line
        cv.line(frame, pt1, pt2, (0, 255, 0), 5)
    return frame
def load_model():
   model = YOLO("yolo11n.pt")
   return model
def analyze_with_model(model, tracker, frame, tracked_ids=None, tracked_items=None):
    if tracked_ids is None:
        tracked_ids = []
    if tracked_items is None:
        tracked_items = {}

    results = model(frame)
    result = results[0] 
    items_found = []
    names = result.names

    for r in result.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = r
        class_name = names[int(class_id)]
        items_found.append([int(x1), int(y1), int(x2), int(y2), float(score), int(class_id), class_name])

    tracker.update(frame, items_found) 

    for track in tracker.tracks:
        track_id = track.track_id
        class_id = track.class_id
        class_name = track.class_name

        tracked_items[track_id] = (class_id, class_name)

        if track_id not in tracked_ids:
            tracked_ids.append(track_id)

    frame = draw_bounding_boxes(frame, tracker.tracks)
    return frame
# def output_compact_class(class_id, class_name, bounding_box):
#     item = (class_name, class_id, bounding_box)
#     return item
def draw_bounding_boxes(frame, trackers):
    class_counts = Counter([t.class_name for t in trackers if t.class_name is not None])
    
    for item in trackers:
        x1, y1, x2, y2 = map(int, item.bbox)
        font = cv.FONT_HERSHEY_SIMPLEX
        # text = f"{item.class_name}, id: {item.class_id}"
        # # print(text)
        # cv.putText(frame, text, (x1, y1 - 10), font, 0.8, (255, 255, 255), 2, cv.LINE_AA)
        color = item.bb_color

        cv.rectangle(frame,(x1,y1),(x2, y2),color,3)
        font = cv.FONT_HERSHEY_SIMPLEX
        # label = f"{item.class_name}, ID: {item.track_id}"
        label = f"{item.track_id}"

        cv.putText(frame, label, (x1, y1 - 10), font, 0.6, (255, 255, 255), 2)
        cv.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    # y_offset = 100
    # # text = "Currenlty on Screen"
    # # cv.putText(frame, text, (10, y_offset), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    # # for class_name, count in class_counts.items():
    # #     text = f"{class_name}: {count}"
    # #     cv.putText(frame, text, (10, y_offset), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    # #     y_offset += 25
    # # return frame
    # # cv.imshow("Tracked Items", frame)
    # # cv.waitKey(1)
    return frame
def is_center_inside(bbox, rect):
    margin = 1
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    rx1, ry1, rx2, ry2 = rect
    
    return (rx1 - margin) <= cx <= (rx2 + margin) and \
           (ry1 - margin) <= cy <= (ry2 + margin)

def track_crosses(tracks, threshold_rect):
    for item in tracks:
        if item.class_name is None:
            continue  # skip if not labeled

        class_name = item.class_name
        track_id = item.track_id

        if is_center_inside(item.bbox, threshold_rect) and class_name in VALID_CLASSES:
            if item.score > 0.71:
                if class_name not in class_entry_counts:
                    class_entry_counts[class_name] = 0
                    seen_ids_by_class[class_name] = set()

                if track_id not in seen_ids_by_class[class_name]:
                    x1, y1, x2, y2 = map(int, item.bbox)
                    width = x2 - x1
                    height = y2 - y1
                    
                    if class_name == "person":
                        score = item.score
        
                        people_dimensions.append([width, height, score])
                        print(f"{class_name} entered, bbox size: {width}x{height}")
                        if height > 100:
                            class_entry_counts[class_name] += 1
                    elif class_name == "car":
                        score = item.score
        
                        car_dimensions.append([track_id, width, height, score])
                        print(f"{class_name} entered, bbox size: {width}x{height}")
                        class_entry_counts[class_name] += 1

                    else:
                        class_entry_counts[class_name] += 1

                    

                    seen_ids_by_class[class_name].add(track_id)
def draw_counts(frame):
    y_offset = 50

    for class_name, count in class_entry_counts.items():
        text = f"{class_name}: {count}"
        cv.putText(frame, text, (10, y_offset), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        y_offset += 25
    return frame
def main():
    file = "TrafficVideo.mp4"
    frames = load_video_file(file)
    # img = load_images(["single_frame.png"])
    # cv.imwrite("single_frame.png", frames[0])
    # lines(img)
    tracker = Tracker()
    model = load_model()
    modified_frames = []
    valid_rectangle = np.array([[235,425], [600,300], [900,275],[785,500]])
    x, y, w, h = cv.boundingRect(valid_rectangle.reshape((-1, 1, 2)))
    threshold_rect = (x, y, x + w, y + h)
    for frame in tqdm(frames, desc="Processing Frames"):
        good_lines = lines(frame)
        frame = analyze_with_model(model,tracker, frame)
        
        track_crosses(tracker.tracks, threshold_rect)  
        frame = draw_counts(frame)
        frame = draw_lines(frame, good_lines)
        modified_frames.append(frame)
    for car in car_dimensions:
        print(car)
    write_video(modified_frames, "tracked_motions.mp4")
    playback_video("tracked_motions.mp4") 
    # draw_bounding_boxes(img, items)
    # play_back_video(frames)
    
if __name__ == "__main__":
    main()