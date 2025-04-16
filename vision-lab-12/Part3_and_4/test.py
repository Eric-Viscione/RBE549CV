import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import os
# flowers ={
# 0: "daisy",
# 1: "dandelion",
# 2: "roses",
# 3: "sunflowers",
# 4: "tulips"
# }
# img_height = 180
# img_width = 180
# TF_MODEL_FILE_PATH = 'Part3/model.tflite'

# # Download example image
# sunflower_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/592px-Red_sunflower.jpg"
# sunflower_path = tf.keras.utils.get_file('Red_sunflower', origin=sunflower_url)

# # Load and preprocess image
# img = tf.keras.utils.load_img(sunflower_path, target_size=(img_height, img_width))
# img_array = tf.keras.utils.img_to_array(img)
# # img_array = img_array / 255.0  # Normalize if model used Rescaling(1./255)
# img_array = np.expand_dims(img_array, axis=0)  # Shape: (1, 180, 180, 3)

# # Load TFLite model
# interpreter = tf.lite.Interpreter(model_path=TF_MODEL_FILE_PATH)
# interpreter.allocate_tensors()
# classify_lite = interpreter.get_signature_runner('serving_default')

# # Make prediction using correct input name
# predictions_lite = classify_lite(keras_tensor_4=img_array)['output_0']
# scores = tf.nn.softmax(predictions_lite)

# print("Prediction probabilities:", scores)
# print("Predicted class:", flowers[np.argmax(scores)])

print(os.path.exists('Part3_and_4/digits_model.tflite'))
DIGITS_TF_MODEL_FILE_PATH = 'Part3_and_4/digits_model.tflite' # The default path to the saved TensorFlow Lite model

interpreter = tf.lite.Interpreter(model_path=DIGITS_TF_MODEL_FILE_PATH)
interpreter.allocate_tensors()

# Get input names
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Print the input and output names
print("Input names:", [input['name'] for input in input_details])
print("Output names:", [output['name'] for output in output_details])