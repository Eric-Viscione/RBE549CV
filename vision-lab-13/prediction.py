import numpy as np
import cv2 as cv
import datetime
import os
from tkinter import *
from PIL import Image, ImageTk
from pynput import keyboard
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import os
import tensorflow as tf
# Load compressed models from tensorflow_hub
os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'
import IPython.display as display

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (12, 12)
mpl.rcParams['axes.grid'] = False

import numpy as np
import PIL.Image
import time
import functools

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
        self.predict_flower = False
        self.predict_digits = False
        self.stylized = False
        self.save_image_bool = False
        self.static_image = False
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

    def start_run(self):
        self.create_directory()
        self.create_directory('temp')
        cv.namedWindow('EV_Capture')
    def evaluate_model(self, frame, model, type):
        flowers = {
            0: "daisy",
            1: "dandelion",
            2: "roses",
            3: "sunflowers",
            4: "tulips"
        }

       

        classify_lite = model.get_signature_runner('serving_default')

        if type == "Flower":
            img_height = 180
            img_width = 180 
            img = cv.resize(frame, (img_width, img_height))
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            img_array = tf.keras.utils.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            predictions_lite = classify_lite(keras_tensor_4=img_array)['output_0']
            scores = tf.nn.softmax(predictions_lite)
            print("Predicted flower:", flowers[np.argmax(scores)])
            
        else:  # "Digit"
            img = cv.resize(frame, (28, 28))                 # Resize to 28x28
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)        # Convert to grayscale
            img_array = tf.keras.utils.img_to_array(img)     # shape: (28, 28, 1)
            img_array = img_array / 255.0                    # Normalize if needed
            img_array = np.expand_dims(img_array, axis=0)    # shape: (1, 28, 28, 1)
            predictions_lite = classify_lite(keras_tensor=img_array)['output_0']
            scores = tf.nn.softmax(predictions_lite)
            print("Predicted digit:", np.argmax(scores))
    def load_img(self,path_to_img):
        max_dim = 512
        img = tf.io.read_file(path_to_img)
        img = tf.image.decode_image(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)

        shape = tf.cast(tf.shape(img)[:-1], tf.float32)
        long_dim = max(shape)
        scale = max_dim / long_dim

        new_shape = tf.cast(shape * scale, tf.int32)

        img = tf.image.resize(img, new_shape)
        img = img[tf.newaxis, :]
        return img
    def preprocess_frame(self, frame, target_size=(256, 256)):
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        tensor = tf.convert_to_tensor(frame_rgb, dtype=tf.float32) / 255.0
        tensor = tf.image.resize(tensor, target_size)
        return tf.expand_dims(tensor, 0)
    def tensor_to_image(self, tensor):
        tensor = tensor * 255.0
        tensor = tf.cast(tensor, tf.uint8)

        if len(tensor.shape) == 4:
            tensor = tf.squeeze(tensor, axis=0)  # Remove batch dimension

        np_img = tensor.numpy()  # TensorFlow tensor ➜ NumPy array
        bgr_img = cv.cvtColor(np_img, cv.COLOR_RGB2BGR)  # RGB ➜ BGR for OpenCV
        return bgr_img
    def stylize_image(self, frame, hub_model, style_image):
        content_image = self.preprocess_frame(frame)
        stylized_image = hub_model(tf.constant(content_image), tf.constant(style_image))[0]
        output_image = self.tensor_to_image(stylized_image)
        return output_image
    def actions(self, frame, action):
        """parser to set the status of togglable actions

        Args:
            frame (_type_): _description_
            action (int): the key pressed
            
        Returns:
            _type_: _description_
        """   

        # for key, (action_func, action_name) in key_actions.items():
        #     if key in self.pressed_keys:
        #         action_func = not action_func
        #         self.toggle_capture(action_func, action_name)
        #         self.pressed_keys.remove(key)
        if 'c' in self.pressed_keys:
            frame = self.save_image(frame)
            self.pressed_keys.remove('c')
            return frame
        if 'f' in self.pressed_keys:
            self.predict_flower = not self.predict_flower
            self.toggle_action(self.predict_flower, "Prediction Flower")
            self.pressed_keys.remove('f')
        if 'd' in self.pressed_keys:
            self.predict_digits = not self.predict_digits
            self.toggle_action(self.predict_digits, "Prediction Digit")
            self.pressed_keys.remove('d')
        if 'm' in self.pressed_keys:
            self.stylized = not self.stylized
            self.toggle_action(self.stylized, "Prediction Digit")
            self.pressed_keys.remove('m')
        # elif action == ord('v'):


        elif action == 27: #escape key
            self.running = False
        return frame


    def main_run(self):
        self.start_run()
        # FLOWER_TF_MODEL_FILE_PATH = 'model.tflite' # The default path to the saved TensorFlow Lite model
        # flower_interpreter = tf.lite.Interpreter(model_path=FLOWER_TF_MODEL_FILE_PATH)
        # DIGITS_TF_MODEL_FILE_PATH = 'digits_model.tflite' # The default path to the saved TensorFlow Lite model
        # digits_interpreter = tf.lite.Interpreter(model_path=DIGITS_TF_MODEL_FILE_PATH)
        # flower_interpreter.allocate_tensors()
        # digits_interpreter.allocate_tensors()
        hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
        style_image = self.load_img("style.jpeg")
        while self.running:

            ret , frame = self.cap.read()
            if not ret:
                print("Can't recieve frame (stream end?). Exiting .....")
                break
            # frame = self.modifiers.static_modifiers(frame, self.font, True, False) ##optional inputs after font to toggle on and off static filters
            # print(self.controls)

            action = cv.waitKey(1)
            frame = self.actions(frame, action)
            # if self.predict_flower:
            #     flower_result = self.evaluate_model(frame, flower_interpreter, "Flower")
            #     self.predict_flower = False
            # if self.predict_digits:
            #     digit_ = self.evaluate_model(frame, digits_interpreter, "Digit")
            #     self.predict_digits = False
            if self.stylized:
                frame = self.stylize_image(frame, hub_model, style_image)
                self.predict_digits = False
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
    video = vid_capture(modifier)
    video.main_run()

if __name__ == "__main__":
    main()


