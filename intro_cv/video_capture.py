import numpy as np
import cv2 as cv
import datetime
import os
from tkinter import *
from PIL import Image, ImageTk


class vid_capture:
    def __init__(self):
        self.cap = cv.VideoCapture(0)
        self.fourcc = cv.VideoWriter_fourcc(*'XVID')
        self.capture_video = False
        self.out = None
        if not self.cap.isOpened():
            print("Cannot open Camera")
            exit()
        self.directory = 'Captures'
    def create_directory(self):
        try: ##directory creation references from geeksforgeeks.com
            os.mkdir(self.directory)
            print(f"Directory '{self.directory}' created successfully.")
        except FileExistsError:
            print(f"Directory '{self.directory}' already exists.")
        except PermissionError:
            print(f"Permission denied: Unable to create '{self.directory}'.")
        except Exception as e:
            print(f"An error occurred: {e}")
    def save_image(self,frame):
        print("Captured Image")
        filepath = f"{self.directory}/{self.generate_name("image", "jpg")}"
        cv.imwrite(filepath, frame) 
    def toggle_capture(self):
        """Start or stop video recording."""
        if not self.capture_video:
            filepath = f"{self.directory}/{self.generate_name("video", "avi")}"

            self.out = cv.VideoWriter(filepath, self.fourcc, 20.0, (640, 480))
            print(f"Started video capture: ")
        else:
            print("Stopped video capture.")
            self.out.release()
            self.out = None
        self.capture_video = not self.capture_video
    def modifiers(self, frame):   ##add all the text and needed transforms to the image
        # cv.line(frame,(0,0),(511,511),(255,0,0),5)
        font = cv.FONT_HERSHEY_SIMPLEX 
        current_time = datetime.datetime.now()
        date_text = f"{current_time.year}/{current_time.month}/{current_time.day}  {current_time.hour}:{current_time.minute}"
        cv.putText(frame, date_text, (300, 450), font, 1, (0, 0, 0), 2)
        cv.rectangle(frame,(2,2),(325,75),(255,255,255),-1) 
        cv.putText(frame, "Controls:", (15,15), font, 0.5, (0,0,0), 2)
        close = "esc: Close the program"
        image = "c: Capture an image"
        video = "v: start and end recording of video"
        controls = [close, image, video]
        for i in range(3):
            cv.putText(frame,controls[i], (30, 30+i*15), font, 0.5, (0,0,0), 2 )
        
        
                

        return frame
    def zoom(self, frame):
        zoom_factor = cv.getTrackbarPos('Zoom', 'EV_Capture')   
        # print(f"Zoom factor is {zoom_factor}")
        if zoom_factor < 1:
            zoom_factor = 1
            # print("Correcting zoom factor")
        zoom_factor = zoom_factor/100.0
        height, width = frame.shape[:2] ##get teh shape of the image
        height_center = height //2
        width_center = width //2
        zoom_height_center_1 = int(height_center- height // (2 * zoom_factor))
        zoom_height_center_2 = int(height_center + height // (2 * zoom_factor))
        zoom_width_center_1 = int(width_center - width // (2 * zoom_factor))
        zoom_width_center_2 = int(width_center + width_center // (2 * zoom_factor))
        zoom_height_center_1 = max(0, zoom_height_center_1)
        zoom_height_center_2 = min(height, zoom_height_center_2)
        zoom_width_center_1 = max(0, zoom_width_center_1)
        zoom_width_center_2 = min(width, zoom_width_center_2)

        frame_region = frame[zoom_height_center_1:zoom_height_center_2, zoom_width_center_1:zoom_width_center_2]  
        new_frame = cv.resize(frame_region, (width, height))
        return new_frame
    def generate_name(self, prefix, file_type):
        filename = f"{prefix}-{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}.{file_type}"
        return filename
    def main_run(self):
        self.create_directory()
        cv.namedWindow('EV_Capture')
        cv.createTrackbar('Zoom','EV_Capture',1,255,lambda x: None)

        while True:
            ret , frame = self.cap.read()
            if not ret:
                print("Can't recieve frame (stream end?). Exiting .....")
                break
            
            frame = self.modifiers(frame)
            frame = self.zoom(frame)
            cv.imshow('frame', frame)
            action = cv.waitKey(1)
            if action == ord('c'):
                self.save_image(frame)
            elif action == ord('v'):
                self.toggle_capture()
            elif action == 27: #escape key
                break
            if self.capture_video and self.out:
                self.out.write(frame)
        self.end()

    def end(self):
        self.cap.release()
        if self.out is not None:
            self.out.release()
        cv.destroyAllWindows()


def main():
    video = vid_capture()
    video.main_run()

if __name__ == "__main__":
    main()



