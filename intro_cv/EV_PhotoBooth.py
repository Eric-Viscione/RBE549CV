import numpy as np
import cv2 as cv
import datetime
import os
from tkinter import *
from PIL import Image, ImageTk



class vid_modifiers:
    def __init__(self):
        pass

    def static_modifiers(self, frame, font, timestamp_toggle = True, controls_toggle = True, border_toggle = True, add_logo_toggle = True):   ##add all the text and needed transforms to the image
        # cv.line(frame,(0,0),(511,511),(255,0,0),5)
        if timestamp_toggle:
            frame = self.timestamp(frame, font)
        if controls_toggle:
            frame = self.controls(frame, font)
        if border_toggle:
            frame = self.border(frame)
        if add_logo_toggle:
            frame = self.add_logo(frame, filepath = 'opencv_logo.png')
        return frame

    def controls(self, frame, font):
        cv.putText(frame, "Controls:", (15,15), font, 0.5, (0,0,0), 2)
        controls = ["esc: Close the program", "c: Capture an image", "v: start and end recording of video"]
        for i, text in enumerate(controls):
            cv.putText(frame, text, (30, 30+i*15), font, 0.5, (0,0,0), 2 )
        cv.rectangle(frame,(2,2),(325,75),(255,255,255),-1) 
        return frame

    def timestamp(self, frame, font):
        current_time = datetime.datetime.now()
        date_text = f"{current_time.year}/{current_time.month}/{current_time.day}  {current_time.hour}:{current_time.minute}"
        cv.rectangle(frame,(300,480),(640,450),(255,255,255),-1) 
        cv.putText(frame, date_text, (300, 475), font, 1, (0, 0, 0), 2)
        return frame
    
    def border(self, frame, color = [0, 0, 255], width = 10):
        return cv.copyMakeBorder(frame, width, width,width, width, cv.BORDER_CONSTANT, value=color)
    
    def add_logo(self, frame, filepath, opac_1 = 0.8, opac_2 = 0.5):
        frame_copy = frame.copy()
        logo = cv.imread(filepath)       #read in image
        assert logo is not None, "file could not be read, check with os.path.exists()"
        mask = self.apply_threshold(logo, 5, 255) #apply the threshold to the mask and return back the logo itself with black everywhere else
        
        img2_fg = cv.resize(cv.bitwise_and(logo,logo,mask = mask), (100,100))
        rows,cols,_ = img2_fg.shape
        frame[0:rows, 0:cols] = img2_fg
        blended_frame = cv.addWeighted(frame_copy, opac_1, frame, 1-opac_1, 0)
        return blended_frame


    def zoom(self, frame, zoom_factor):
        if zoom_factor < 1: #corrects the zoom factor so we never divide by 0 
            zoom_factor = 1
        zoom_factor = zoom_factor/100.0
        height, width = frame.shape[:2] ##get the shape of the image
        height_center,width_center = height //2, width //2
        w = int(width / (2 * zoom_factor))
        h = int(height / (2 * zoom_factor))
        frame_region = frame[height_center - h:height_center + h, width_center - w:width_center + w]
        return cv.resize(frame_region, (width, height))
    
    def extract_color(self, frame, color_range = []):
        color_range = [np.array([110,50,50]),np.array([130,255,255])]
        hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsv_frame, color_range[0], color_range[1])
        return cv.bitwise_and(frame,frame,mask = mask)
    
    def rotate_image(self, frame, angle = 10):   #default 10 deg rotation
        rows, cols = frame.shape[:2]
        image_center = (cols-1)/2.0, (rows-1)/2.0
        M = cv.getRotationMatrix2D(image_center, angle, 1)
        return cv.warpAffine(frame, M, (cols, rows))

    def apply_threshold(self, frame, lower_threshold = 127, upper_threshold = 255,  threshold_type = cv.THRESH_BINARY):
        grayscale = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        _, new_frame = cv.threshold(grayscale,lower_threshold, upper_threshold, threshold_type)
        return new_frame
    
    def gaussian_blur(self, frame, kernel = 5, x = None, y = None):
        if kernel % 2 == 0:
            kernel += 1  # keep the kernel odd
        if x == None: x = max(5, min(cv.getTrackbarPos('Gaussian SigmaX', 'EV_Capture'), 30))
        if y == None: y = max(5, min(cv.getTrackbarPos('Gaussian SigmaY', 'EV_Capture'), 30))
        blur = cv.GaussianBlur(frame, (kernel,kernel), x, y)
        return blur
    
    def sharpen_image(self, frame, alpha = 0.75):
        alpha = cv.getTrackbarPos('Sharpening Alpha', 'EV_Capture')/20
        kernel = cv.getTrackbarPos('Kernel Size', 'EV_Capture')
        blurred_image = self.gaussian_blur(frame, kernel)
        detail = cv.subtract(frame, blurred_image)
        detail = alpha*detail
        detail = detail.astype(frame.dtype)
        new_frame = cv.add(frame, detail)
        return new_frame
    
    def copy_roi(self, frame):
        roi = frame[455:485,300:620] #rows, cols
        frame[5:35,320:640] = roi
        return frame


class vid_capture:
    def __init__(self, modifiers):
        self.cap = cv.VideoCapture(0)
        self.fourcc = cv.VideoWriter_fourcc(*'XVID')
        self.capture_video = False
        self.out = None
        if not self.cap.isOpened():
            print("Cannot open Camera")
            exit()
        self.directory = 'Captures'
        self.font = cv.FONT_HERSHEY_SIMPLEX
        self.running = True  
        self.modifiers = modifiers
        self.extract = False
        self.rotate = False
        self.threshold = False
        self.copy_roi = False
        self.gaussian_blur = False
        self.sharpen = False

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

    def save_image(self,frame, color = []):
        print("Captured Image")
        # rows, cols, channels = frame.shape
        filepath = f"{self.directory}/{self.generate_name('image', 'jpg')}"
        cv.imwrite(filepath, frame) 
        frame[:] = [255, 255, 255] #flash white
        print(f"Frames shape is{frame.shape}")
        return frame

    def toggle_capture(self):
        """Start or stop video recording."""
        if not self.capture_video:
            filepath = f"{self.directory}/{self.generate_name('video', 'avi')}"

            self.out = cv.VideoWriter(filepath, self.fourcc, 20.0, (640, 480))
            print(f"Started video capture: ")
        else:
            print("Stopped video capture.")
            self.out.release()
            self.out = None
        self.capture_video = not self.capture_video

    def toggle_action(self,flag, action = []):
        print(F"{action}")
        if flag:
            print(f"Starting {action}:")
        else:
            print(f"Stopping {action}.")

    def generate_name(self, prefix, file_type): #genereates the name for our images and captures
        filename = f"{prefix}-{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}.{file_type}"
        return filename
        
    def start_run(self):
        self.create_directory()
        cv.namedWindow('EV_Capture')
        cv.createTrackbar('Zoom','EV_Capture',1,255,lambda x: None)
        cv.createTrackbar('Gaussian SigmaX','EV_Capture',5,30,lambda x: None)
        cv.createTrackbar('Gaussian SigmaY','EV_Capture',5,30,lambda x: None)
        cv.createTrackbar('Sharpening Alpha','EV_Capture',0,100,lambda x: None)
        cv.createTrackbar('Kernel','EV_Capture',0,20,lambda x: None)

    def actions(self, frame, action, color = []):
            if action == ord('c'):
                frame = self.save_image(frame)
                return frame
            elif action == ord('v'):
                self.toggle_capture()
            elif action == ord('e'):
                self.extract = not self.extract
                self.toggle_action(self.extract, "Extracting Color")
            elif action == ord('r'):
                self.rotate = not self.rotate
                self.toggle_action(self.rotate, "Rotating Image")
            elif action == ord('t'):
                self.threshold = not self.threshold
                self.toggle_action(self.threshold, "Applying Threshold")
            elif action == ord('b'):
                self.gaussian_blur = not self.gaussian_blur
                self.toggle_action(self.gaussian_blur, "Gaussian Blurring")
            elif action == ord('s'):
                self.sharpen = not self.sharpen
                self.toggle_action(self.sharpen, "Sharpening")
            elif action == ord('m'):
                self.copy_roi = not self.copy_roi
                self.toggle_action(self.copy_roi, "Copying ROI")
            elif action == 27: #escape key
                self.running = False
            return frame


    def main_run(self):
        self.start_run()
        while self.running:
            ret , frame = self.cap.read()
            if not ret:
                print("Can't recieve frame (stream end?). Exiting .....")
                break
            frame = self.modifiers.static_modifiers(frame, self.font, True, False) ##optional inputs after font to toggle on and off static filters
            zoom_factor = cv.getTrackbarPos('Zoom', 'EV_Capture') 
            frame = self.modifiers.zoom(frame, zoom_factor)
            action = cv.waitKey(1)
            

            frame = self.actions(frame, action)
            if self.extract:
                frame = self.modifiers.extract_color(frame) #optionally define color as a tuple of arrays color = [np.array([lower_r,lower_g,lower_b]),np.array([upper_r,upper_g,upper_b])]
            if self.rotate:
                frame = self.modifiers.rotate_image(frame)
            if self.threshold:
                frame = self.modifiers.apply_threshold(frame) #optional arguments for the lower and upper threshold bounds adn threshold type eg apply_threshold(frame, lwoer, upper, type)
            if self.copy_roi:
                frame = self.modifiers.copy_roi(frame)
            if self.gaussian_blur:
                frame = self.modifiers.gaussian_blur(frame)
            if self.sharpen:
                frame = self.modifiers.sharpen_image(frame)
            cv.imshow('frame', frame)
            if self.capture_video and self.out:
                self.out.write(frame)
            
        self.end()

    def end(self):
        self.cap.release()
        if self.out is not None:
            self.out.release()
        cv.destroyAllWindows()


def main():
    modifier = vid_modifiers()
    video = vid_capture(modifier)
    video.main_run()

if __name__ == "__main__":
    main()



