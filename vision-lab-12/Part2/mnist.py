import tensorflow as tf
import tensorboard
import keras_preprocessing
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator
import keras
import datetime
from datetime import datetime

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
# Define the Keras TensorBoard callback.
logdir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)


model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation='softmax')
])


loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
model.fit(
    x_train,
    y_train,
    batch_size=64,
    epochs=5,
    validation_data=(x_test, y_test),
    callbacks=[tensorboard_callback]
)

model.evaluate(x_test,  y_test, verbose=2)

model.save("digits.h5")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model.
with open('digits_model.tflite', 'wb') as f:
  f.write(tflite_model)