import numpy as np
import cv2 as cv
import datetime
import os
from tkinter import *
from PIL import Image, ImageTk
from pynput import keyboard
from matplotlib import pyplot as plt
import math
from shared_libraries import feature_matching
from time import sleep
import sys

def panorama(cap, num_images = 15):
    panorama_images = []
    while len(panorama_images) < num_images:
        ret, frame = cap.read()
        
        if not ret:
            print("Can't receive frame (stream end?). Exiting .....")
            return False, None
        
        panorama_images.append(frame)
        
        # new_frame=cv.putText(frame, F"pano img {len(panorama_images)}:", (100, 100), cv.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)


        cv.imshow(f"pano_image {len(panorama_images)}", frame)
        key = cv.waitKey(250)
        print(f"Captured image {len(panorama_images)}")
    cv.destroyAllWindows()
    
    # panorama_images = [cv.imread('boston1.jpeg'), cv.imread('boston2.jpeg')]
    stitcher = cv.Stitcher.create()
    status, pano = stitcher.stitch(panorama_images)
    if status != cv.Stitcher_OK:
        print("Can't stitch images, error code = %d" % status)
        sys.exit(-1)
        return  False, None
    pano = cv.resize(pano, (1800, 600))
    cv.imshow("Panorama", pano)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
    return True, pano

class Image_classifiers:
    def __init__(self):
        pass
    def harris(self, img, blocksize=2, ksize=3, k=0.04, thresh=0.05):
        # print(type(img))
        gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        gray_img = np.float32(gray_img)
        dst = cv.cornerHarris(gray_img,blocksize, ksize, k)
        img_copy = img.copy()
        img_copy[dst > thresh * dst.max()] = [0, 0, 255]
        return img_copy

    def sift(self,img, nfeatures=0, nOctavelayers=3, constant_threshold=0.1, edge_threshold=10, sigma=1.6 ):
        gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        sift = cv.xfeatures2d.SIFT_create(nfeatures=nfeatures, 
                                   nOctaveLayers=nOctavelayers, 
                                   contrastThreshold=constant_threshold, 
                                   edgeThreshold=edge_threshold, 
                                   sigma=sigma)

        keypoints, descriptors = sift.detectAndCompute(gray, None)
        dst = cv.drawKeypoints(gray, keypoints, img, color = (0, 255, 0))
        return dst
    def match_features(self, frame, match_method, extract_method):
        second_image = cv.imread('book-2.jpg')
        second_image = cv.resize(second_image, (600, 600))
        assert second_image is not None, "file to track could not be read, check with os.path.exists()"
        images = [second_image, frame]
        new_frame = feature_matching(images, match_method, extract_method)
        cv.putText(new_frame, f"{extract_method} and {match_method}", (15,15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
        # new_frame = cv.resize(new_frame, (600, 600))
        return new_frame
class vid_modifiers:
    def __init__(self):

        self.font = cv.FONT_HERSHEY_SIMPLEX
        pass

    def static_modifiers(self, frame, timestamp_toggle = True, controls_toggle = False, border_toggle = True, add_logo_toggle = False):   
        """Adds a bunch of modifiers and text to the screen that will be present no matter what

        Args:
            frame (np.array): the frame currently being modifed
      
            timestamp_toggle (bool, optional): Turn on and timestamp in lower right. Defaults to True.
            controls_toggle (bool, optional): Show or dont show the controls panel. Defaults to True.
            border_toggle (bool, optional): Create a red border. Defaults to True.
            add_logo_toggle (bool, optional): Add the open cv logo. Defaults to True.

        Returns:
            np.array: Modified frame
        """
        if timestamp_toggle:
            frame = self.timestamp(frame)
        if border_toggle:
            frame = self.border(frame)
        if add_logo_toggle:
            frame = self.add_logo(frame, filepath = 'opencv_logo.png')
        return frame

    def control_control(self, frame, active = False):
        """Adds a small blurb about the control scheme

        Args:
            frame (np.array): frame currently being modified

        Returns:
            np.array: frame with control explanantion
        """
        if active:
            cv.putText(frame, "p: toggles controls board:", (15,15), self.font, 0.5, (0,0,0), 2)
        return frame
    
    def controls(self, frame):
        """Creates a pop up window with a how to for each instruction

        Args:
            frame (_type_): _description_

        Returns:
            _type_: _description_
        """
        controls = ["esc: Close the program", "c: Capture an image", "v: start and end recording of video",
                    "r: rotate the frame 10 degrees", "t: apply a thresholding filter", "b: Gaussian blur the image",
                     "s: sharpen the image", "m: copy and paste date time", "g + x or y: toggles built in sobel filter",
                    "s + x or y: toggles custom sobel filter", "d: toggles canny filter", "s + l: toggles custom laplacian filter",
                    "p: toggle the controls board", "h: Toggles Harris Detector", "o: Toggles Sift Detector"]
        cv.rectangle(frame,(2,2),(325,75*len(controls)),(255,255,255),-1) 
        cv.putText(frame, "Controls:", (15,15), self.font, 0.5, (0,0,0), 2)

        for i, text in enumerate(controls):
            cv.putText(frame, text, (30, 30 + i*15), self.font, 0.5, (0, 0, 0), 2)
        
        return frame

    def timestamp(self, frame):
        """Adds the time stamp to lower right

        Args:
            frame (_type_): _description_

        Returns:
            _type_: _description_
        """
        current_time = datetime.datetconvultionime.now()
        date_text = f"{current_time.year}/{current_time.month}/{current_time.day}  {current_time.hour}:{current_time.minute}"
        cv.rectangle(frame,(300,480),(640,450),(255,255,255),-1) 
        cv.putText(frame, date_text, (300, 475), self.font, 1, (0, 0, 0), 2)
        return frame
    
    def border(self, frame, color = [0, 0, 255], width = 10):
        """Adds a red border to the frame

        Args:
            frame (_type_): _description_
            color (list, optional): rgb triplet of the color of the border. Defaults to [0, 0, 255].
            width (int, optional): Thickness of the border in pixels. Defaults to 10.

        Returns:
            _type_: _description_
        """
        return cv.copyMakeBorder(frame, width, width,width, width, cv.BORDER_CONSTANT, value=color)
    
    def add_logo(self, frame, filepath, opac_1 = 0.8, opac_2 = 0.5):
        """Adds the opencv logo with opacity to the image

        Args:
            frame (_type_): _description_
            filepath (string): location of the iamge
            opac_1 (float, optional): how much the image should be visible . Defaults to 0.8.
            opac_2 (float, optional): _description_. Defaults to 0.5.

        Returns:
            _type_: _description_
        """
        frame_copy = frame.copy() #creates a proper copy of the image instead of a reference
        logo = cv.imread(filepath)       #read in image
        assert logo is not None, "file could not be read, check with os.path.exists()"
        mask = self.apply_threshold(logo, 5, 255) #apply the threshold to the mask and return back the logo itself with black everywhere else
        # print("here")
        img2_fg = cv.resize(cv.bitwise_and(logo,logo,mask = mask), (100,100))
        rows,cols,_ = img2_fg.shape
        frame[0:rows, 0:cols] = img2_fg
        blended_frame = cv.addWeighted(frame_copy, opac_1, frame, 1-opac_1, 0)
        return blended_frame


    def zoom(self, frame, zoom_factor):
        """Controls how zoomed in the image is

        Args:
            frame (_type_): _description_
            zoom_factor (int): Zoom amount gotten by the scroll bar

        Returns:
            _type_: _description_
        """
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
        """extracts a color from the image

        Args:
            frame (_type_): _description_
            color_range (list, optional): tuple of triplets that describe the range of colors that should be found. Defaults to [].

        Returns:
            _type_: _description_
        """
        color_range = [np.array([110,50,50]),np.array([130,255,255])]
        hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsv_frame, color_range[0], color_range[1])
        return cv.bitwise_and(frame,frame,mask = mask)
    
    def rotate_image(self, frame, angle = 10):   #default 10 deg rotation
        """Rotates the entire frame by angle degrees

        Args:
            frame (_type_): _description_
            angle (int, optional): how many degrees to rotate the image. Defaults to 10.

        Returns:
            _type_: _description_
        """
        rows, cols = frame.shape[:2]
        image_center = (cols-1)/2.0, (rows-1)/2.0
        M = cv.getRotationMatrix2D(image_center, angle, 1)
        return cv.warpAffine(frame, M, (cols, rows))

    def apply_threshold(self, frame, lower_threshold = 127, upper_threshold = 255,  threshold_type = cv.THRESH_BINARY):
        """Applies a thresholding filter that turns everything above 127 brightness white and vice versa

        Args:
            frame (_type_): _description_
            lower_threshold (int, optional): value to base the thresholding off of. Defaults to 127.
            upper_threshold (int, optional): max value, needed for calculatation. Defaults to 255.
            threshold_type (_type_, optional): The algorithm used for thresholding. Defaults to cv.THRESH_BINARY.

        Returns:
            _type_: _description_
        """
        grayscale = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        _, new_frame = cv.threshold(grayscale,lower_threshold, upper_threshold, threshold_type)
        return new_frame
    
    def gaussian_blur(self, frame, kernel = 5, x = None, y = None):
        """Applies a gaussian style blurring to the frame

        Args:
            frame (_type_): _description_
            kernel (int, optional): squae size of the kernel. Defaults to 5.
            x (_type_, optional): sigmax value to blur, found with trackbar. Defaults to None.
            y (_type_, optional): sigmay value to blur, flound with trackbar. Defaults to None.

        Returns:
            _type_: _description_
        """
        if kernel % 2 == 0:
            kernel += 1  # keep the kernel odd
        if x == None: x = max(5, min(cv.getTrackbarPos('Gaussian SigmaX', 'EV_Capture'), 30))
        if y == None: y = max(5, min(cv.getTrackbarPos('Gaussian SigmaY', 'EV_Capture'), 30))
        blur = cv.GaussianBlur(frame, (kernel,kernel), x, y)
        return blur
    
    def sharpen_image(self, frame, alpha = 0.75):
        """Sharpen the iamge

        Args:
            frame (_type_): _description_
            alpha (float, optional): The amount of sharpening applied based off of the amount of the details extracted is incorporated. Defaults to 0.75.

        Returns:
            _type_: _description_
        """
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
        frame[5:30,320:640] = roi
        return frame

    def apply_kernel(self, frame, kernel, padding = True, fast_algo = True):
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
    def sobel(self, frame, direction, type):
        """Apply either a cv built in sobel filter to the image or the custom implemented one

        Args:
            frame (_type_): _description_
            direction (_type_): Apply it to either the x or y direction
            type (_type_): Decalres whetehr to use manual or built in one

        Returns:
            _type_: _description_
        """
        kernel = np.array([[-1, 0, 1],
                          [-2, 0, 2],
                          [-1, 0, 1]])
        if len(frame.shape) == 2:  #check if it is already grayscale so we can apply both x and y at the same time
            gray_frame = frame
        else:  
            gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        if direction == 'x':
            kernel = kernel
            if type == 'auto':
                kernel_x = cv.getTrackbarPos('Sobel X Kernel', 'EV_Capture')
                if kernel_x % 2 == 0:
                    kernel_x += 1  # keep the kernel odd
                kernel_x = max(0, min(kernel_x, 30))
                new_frame =  cv.Sobel(gray_frame,cv.CV_64F,1,0,ksize=kernel_x)
        
        elif direction == 'y':
            kernel = np.rot90(kernel)
            if type == 'auto':
                kernel_y = cv.getTrackbarPos('Sobel Y Kernel', 'EV_Capture')
                if kernel_y % 2 == 0:
                    kernel_y += 1  # keep the kernel odd
                kernel_y = max(0, min(kernel_y, 29))
                new_frame =  cv.Sobel(gray_frame,cv.CV_64F,1,0,ksize=kernel_y)
        else:
            print(F"Programming error, Figure out why non valid direction was passed! Recieved {direction}")
            return False
        if type == 'manual':
            new_frame = self.apply_kernel(gray_frame, kernel)
        if type != 'manual' and type != 'auto':
            print(F"Programming error, Figure out why non valid type was passed! Recieved {type}")
            return False
        return new_frame
    def canny(self, frame, kernel = 3):
        """Applies a canny edge detection filter with thresholds defined by track bars

        Args:
            frame (_type_): _description_
            kernel (int, optional): Size of the kernel for the canny filter Defaults to 3.

        Returns:
            _type_: _description_
        """
        threshold_1 = cv.getTrackbarPos('Canny Threshold 1', 'EV_Capture')
        threshold_2 = cv.getTrackbarPos('Canny Threshold 2', 'EV_Capture')
        frame = cv.Canny(frame,threshold_1, threshold_2 )
        return frame
    def combined_sobel(self, frame, type = 'manual'):
        grad_x = self.sobel(frame, 'x', 'manual')
        grad_y = self.sobel(frame, 'y', 'manual')
        abs_grad_x = cv.convertScaleAbs(grad_x)
        abs_grad_y = cv.convertScaleAbs(grad_y)
        grad = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
        return grad
    def laplacian(self, frame):
        """Applies a custom laplacian filter to the image

        Args:
            frame (_type_): _description_

        Returns:
            _type_: _description_
        """
        kernel = np.array([[0, -1, 0],
                          [-1, 4, -1],
                          [0, -1, 0]])
        if len(frame.shape) == 2:  #check if it is already grayscale so we can apply both x and y at the same time
            gray_frame = frame
        else:  
            gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        new_frame = self.apply_kernel(gray_frame, kernel)
        return new_frame

class vid_capture:
    def __init__(self, modifiers, classifiers):
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
        self.classifiers = classifiers
        self.extract = False
        self.rotate = False
        self.threshold = False
        self.copy_roi = False
        self.gaussian_blur = False
        self.sharpen = False
        self.controls = False
        self.auto_sobel_x = False
        self.auto_sobel_y = False
        self.sobel_x = False
        self.sobel_y = False
        self.canny = False
        self.laplacian = False
        self.four_windows = False
        self.save_image_bool = False
        self.static_image = False
        self.harris = False
        self.sift = False
        self.surf_bf = False         
        self.surf_flann     = False  
        self.sift_bf     = False     
        self.sift_flann  = False 
        self.panorama_toggle = False
        self.created_panorama = False
        self.pressed_keys = set()
        self.listener = keyboard.Listener(on_press=self.on_key_press, on_release=self.on_release)
        self.listener.start()
    def on_key_press(self, key, debug = False):
        if debug:
            print(f"The key pressed is {key}")
        try:
            if key.char:
               self.pressed_keys.add(key.char)
        except AttributeError:
            self.pressed_keys.add(key)
    def on_release(self,key, debug = False):
        if key in self.pressed_keys:
            self.pressed_keys.remove(key)
            if debug:
                print(f"Released: {key} - Current keys: {self.pressed_keys}")
        else:
            if debug:
                print(f"Key {key} not found in pressed_keys set.")  
            pass
    def create_directory(self, path = None):
        if path == None:
            path = self.directory
        else:
            path = f'{self.directory}/{path}'
        try: ##directory creation references from geeksforgeeks.com
            os.mkdir(path)
            print(f"Directory '{path}' created successfully.")
        except FileExistsError:
            print(f"Directory '{path}' already exists.")
        except PermissionError:
            print(f"Permission denied: Unable to create '{path}'.")
        except Exception as e:
            print(f"An error occurred: {e}")

    def save_image(self,frame, color = []):
        print("Captured Image")
        # rows, cols, channels = frame.shape
        filepath = f"{self.directory}/{self.generate_name('image', 'jpg')}"
        cv.imwrite(filepath, frame)
        if self.threshold:
            frame[:] = [255]
        else:
            frame[:] = [255, 255, 255] #flash white
        # print(f"Frames shape is{frame.shape}")
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
        """This prints out a statement saying what is being stopped or started in the session

        Args:
            flag (bool): whether it is stopping or starting
            action (list, optional): string of the item so it can be printed. Defaults to [].
        """
        print(F"{action}")
        if flag:
            print(f"Starting {action}:")
        else:
            print(f"Stopping {action}.")

    def generate_name(self, prefix, file_type): #genereates the name for our images and captures
        filename = f"{prefix}-{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}.{file_type}"
        return filename
    
    def stack_images(images, cols = None):
        cols = round(math.sqrt(len(images)))
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


    def toggle_four_windows(self, frame):
        temp_path = f'{self.directory}/temp/temp_image.png'
        laplacian = self.modifiers.laplacian(frame)
        sobel_x = self.modifiers.sobel(frame, 'x', 'manual')
        sobel_y = self.modifiers.sobel(frame, 'y', 'manual')
        plt.subplot(2,2,1),plt.imshow(frame,cmap = 'gray')
        plt.title('Original'), plt.xticks([]), plt.yticks([])
        plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')
        plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
        plt.subplot(2,2,3),plt.imshow(sobel_x,cmap = 'gray')
        plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
        plt.subplot(2,2,4),plt.imshow(sobel_y,cmap = 'gray')
        plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
        plt.savefig(temp_path, bbox_inches='tight', pad_inches=0)
        plt.close()

        img = cv.imread(temp_path)
        os.remove(temp_path)
        return img

    def start_run(self):
        self.create_directory()
        self.create_directory('temp')
        cv.namedWindow('EV_Capture')
        cv.createTrackbar('Zoom',             'EV_Capture', 1, 255, lambda x: None)
        cv.createTrackbar('Gaussian SigmaX',  'EV_Capture', 5, 30,  lambda x: None)
        cv.createTrackbar('Gaussian SigmaY',  'EV_Capture', 5, 30,  lambda x: None)
        cv.createTrackbar('Sharpening Alpha', 'EV_Capture', 0, 100, lambda x: None)
        cv.createTrackbar('Sharpening Kernel','EV_Capture', 0, 20,  lambda x: None)
        cv.createTrackbar('Sobel X Kernel',   'EV_Capture', 0, 30, lambda x: None)
        cv.createTrackbar('Sobel Y Kernel',   'EV_Capture', 0, 30, lambda x: None)
        cv.createTrackbar('Canny Threshold 1',   'EV_Capture', 0, 5000, lambda x: None)
        cv.createTrackbar('Canny Threshold 2',   'EV_Capture', 0, 5000, lambda x: None)

    def actions(self, frame, action):
        """Parser to set the status of togglable actions

        Args:
            frame: The current video frame
            action (int): The key pressed
                
        Returns:
            frame: The modified frame after actions are applied
        """   
        # Define a dictionary mapping keys to (attribute, description) tuples
        key_actions = {
            'c': ('save_image', "Saving Image"),
            'v': ('toggle_capture', "Toggling Capture"),
            'e': ('extract', "Extracting Color"),
            'r': ('rotate', "Rotating Image"),
            't': ('threshold', "Applying Threshold"),
            'b': ('gaussian_blur', "Gaussian Blurring"),
            's': ('sharpen', "Sharpening"),
            'm': ('copy_roi', "Copying ROI"),
            'p': ('controls', "Showing Controls"),
            'd': ('canny', "Apply canny edge detection filter"),
            'h': ('harris', "Turn on Harris Detector"),
            'o': ('sift', "Turn on sift Detector"),
            'l': ('static_image', "Showing static image"),
            '1': ('surf_bf', "Turn on feature matching with brute force and surf"),
            '2': ('surf_flann', "Turn on feature matching with flann and surf"),
            '3': ('sift_bf', "Turn on feature matching with brute force and sift"),
            '4': ('four_windows', "Displaying four windows of Sobel and Laplacian"),
            '5': ('sift_flann', "Turn on feature matching with flann and sift"),
            '6': ('panorama_toggle', "Start Stitching 3 images together")
        }

        # Define key combinations
        key_combinations = {
            frozenset(['g', 'x']): ('auto_sobel_x', "Applying Built in Sobel in X direction"),
            frozenset(['g', 'y']): ('auto_sobel_y', "Applying Built in Sobel in Y direction"),
            frozenset(['s', 'x']): ('sobel_x', "Applying manual Sobel in X direction"),
            frozenset(['s', 'y']): ('sobel_y', "Applying manual Sobel in Y direction"),
            frozenset(['s', 'l']): ('laplacian', "Applying manual Laplacian")
        }

        if 'c' in self.pressed_keys:
            frame = self.save_image(frame)
            self.pressed_keys.remove('c')
            return frame
        
        if 'v' in self.pressed_keys:
            self.toggle_capture()
            self.pressed_keys.remove('v')

        if action == 27:  # escape key
            self.running = False
            return frame

        # Handle single key actions
        for key, (attr_name, description) in key_actions.items():
            if key in self.pressed_keys:
                if key in ['c', 'v']:
                    continue  
                    
                setattr(self, attr_name, not getattr(self, attr_name))
                self.toggle_action(getattr(self, attr_name), description)
                self.pressed_keys.remove(key)

        current_keys = frozenset(self.pressed_keys)
        for combo, (attr_name, description) in key_combinations.items():
            if combo.issubset(current_keys):
                setattr(self, attr_name, not getattr(self, attr_name))
                self.toggle_action(getattr(self, attr_name), description)
                
                for k in combo:
                    if k in self.pressed_keys:
                        self.pressed_keys.remove(k)

        return frame
    def main_run(self):
        modifiers = {
            'extract': (self.modifiers.extract_color, []),
            'rotate': (self.modifiers.rotate_image, []),
            'threshold': (self.modifiers.apply_threshold, []),
            'copy_roi': (self.modifiers.copy_roi, []),
            'gaussian_blur': (self.modifiers.gaussian_blur, []),
            'sharpen': (self.modifiers.sharpen_image, []),
            'controls': (self.modifiers.controls, []),
            'sobel_x': (self.modifiers.sobel, ['x', 'manual']),
            'sobel_y': (self.modifiers.sobel, ['y', 'manual']),
            'auto_sobel_x': (self.modifiers.sobel, ['x', 'auto']),
            'auto_sobel_y': (self.modifiers.sobel, ['y', 'auto']),
            'canny': (self.modifiers.canny, []),
            'laplacian': (self.modifiers.laplacian, [])
        }
    
        classifiers = {
            'harris': (self.classifiers.harris, []),
            'sift': (self.classifiers.sift, []),
            'surf_bf': (self.classifiers.match_features, ['brute_force', 'surf']),
            'surf_flann': (self.classifiers.match_features, ['flann', 'surf']),
            'sift_bf': (self.classifiers.match_features, ['brute_force', 'sift']),
            'sift_flann': (self.classifiers.match_features, ['flann', 'sift'])
        }
    
        self.start_run()
        while self.running:
            if self.static_image:
                frame = cv.imread('sudoku.png')
            else:
                ret , frame = self.cap.read()
                if not ret:
                    print("Can't recieve frame (stream end?). Exiting .....")
                    break
            # print(self.controls)
            if not self.four_windows:
                zoom_factor = cv.getTrackbarPos('Zoom', 'EV_Capture') 
                frame = self.modifiers.zoom(frame, zoom_factor)
                frame = self.modifiers.static_modifiers(frame, self.font, True, False) ##optional inputs after font to toggle on and off static filters
                for attr_name, (func, args) in modifiers.items():
                    if getattr(self, attr_name, False):
                        frame = func(frame, *args)
                for attr_name, (func, args) in classifiers.items():
                    if getattr(self, attr_name, False):
                        frame = func(frame, *args)
                if not self.controls:
                    frame = self.modifiers.control_control(frame)
                else:
                    frame = self.toggle_four_windows(frame)
        
                if self.panorama_toggle:
                    self.panorama_toggle = False
                    _, frame = panorama(self.cap)
                    print("pano")
            action = cv.waitKey(1)
            frame = self.actions(frame, action)
            cv.imshow('frame', frame)
            if self.capture_video and self.out:
                self.out.write(frame)
        self.end()

    def end(self):
        """Destroys any frames remaining and frees them
        """
        self.cap.release()
        if self.out is not None:
            self.out.release()
        cv.destroyAllWindows()


def main():
    modifier = vid_modifiers()
    classifier = Image_classifiers()
    video = vid_capture(modifier, classifier)
    
    video.main_run()

if __name__ == "__main__":
    main()



