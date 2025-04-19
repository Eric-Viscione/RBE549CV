import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import kagglehub
import pandas as pd

from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input as vgg19_preprocess
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input as inceptionv3_preprocess

BATCH_SIZE = 32
IMG_SIZE = (220, 220)


# path = kagglehub.dataset_download("agrigorev/clothing-dataset-full")

# # print("Path to dataset files:", path)
# path = kagglehub.dataset_download("andhikawb/fashion-mnist-png")

# print("Path to dataset files:", path)

# exit()
# label_map = {
#     0: "T-shirt/top",
#     1: "Trouser",
#     2: "Pullover",
#     3: "Dress",
#     4: "Coat",
#     5: "Sandal",
#     6: "Shirt",
#     7: "Sneaker",
#     8: "Bag",
#     9: "Ankle boot"
# }

# train_dir = "/home/mainubuntu/.cache/kagglehub/datasets/andhikawb/fashion-mnist-png/versions/1/train"
# validation_dir = "/home/mainubuntu/.cache/kagglehub/datasets/andhikawb/fashion-mnist-png/versions/1/test"
dataset_path = 'MerchData'
# dataset = tf.keras.utils.image_dataset_from_directory(
#     dataset_path,
#     image_size=IMG_SIZE,
#     batch_size=BATCH_SIZE,
#     validation_split=0.2,         
#     subset="training",             
#     seed=123                       
# )
train_dataset = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)
class_names = train_dataset.class_names
# class_names = [label_map[int(idx)] for idx in train_dataset.class_names]
# df = pd.read_csv(class_dir)
# class_names = sorted(df['label'].unique())

val_batches = tf.data.experimental.cardinality(validation_dataset)
validation_dataset = validation_dataset.shuffle(buffer_size=1000, seed=123)
# test_dataset = validation_dataset.take(val_batches // 5)
# validation_dataset = validation_dataset.skip(val_batches // 5)

data_augmentation = tf.keras.Sequential([
  tf.keras.layers.RandomFlip('horizontal'),
  tf.keras.layers.RandomRotation(0.2),
])


preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
IMG_SHAPE = IMG_SIZE + (3,)

# vgg19_model = 
models = {
    "vgg19":{
        "model": tf.keras.applications.VGG19(include_top=False,weights="imagenet",input_shape=IMG_SHAPE, pooling='avg' ),
        "preprocess": vgg19_preprocess
    },
    # "inceptionv3":{
        # "model": InceptionV3(include_top=False, weights="imagenet", input_shape=IMG_SHAPE, pooling='avg'),
        # "preprocess": inceptionv3_preprocess
# 
    # }
}
for model_name, config in models.items():
    base_model = config["model"]
    current_preprocess = config["preprocess"] 

    # image_batch, label_batch = next(iter(train_dataset))
    # feature_batch = base_model(image_batch)
    # print(feature_batch.shape)

    base_model.trainable = False
    num_classes = len(class_names)
    loss='sparse_categorical_crossentropy'
    metrics=['accuracy']
    # global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    # feature_batch_average = global_average_layer(feature_batch)

    prediction_layer = tf.keras.layers.Dense(num_classes, activation='softmax')
    # prediction_batch = prediction_layer(feature_batch)

    inputs = tf.keras.Input(shape=IMG_SHAPE)
    x = data_augmentation(inputs)
    x = current_preprocess(x)
    x = base_model(x, training=False)
    # x = global_average_layer(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = prediction_layer(x)
    model = tf.keras.Model(inputs, outputs)
    # tf.keras.utils.plot_model(model, show_shapes=True)
    base_learning_rate = 0.0001
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    initial_epochs = 10

    loss0, accuracy0 = model.evaluate(validation_dataset)
    history = model.fit(train_dataset,
                        epochs=initial_epochs,
                        validation_data=validation_dataset)
    loss0, accuracy0 = model.evaluate(validation_dataset)
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    model.save(model_name)
    plt.figure(figsize=(8, 8))

    # Accuracy Plot
    plt.subplot(2, 1, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.title('Accuracy Over Epochs')
    plt.legend()

    # Loss Plot
    plt.subplot(2, 1, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.title('Loss Over Epochs')
    plt.legend()

    plt.tight_layout()
    plt.show()

