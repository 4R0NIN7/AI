import numpy as np
from absl import logging
from PIL import Image
from tflite_model_maker.object_detector import DataLoader

import tensorflow as tf

assert tf.__version__.startswith("2")

from utils.create_model import create_model
from utils.detect_on_image import detect_on_image

tf.get_logger().setLevel("ERROR")
logging.set_verbosity(logging.DEBUG)


# Splitting data
train_data = DataLoader.from_pascal_voc(
    "data/android_figurine/train",
    "data/android_figurine/train",
    ["android", "pig_android"],
)
val_data = DataLoader.from_pascal_voc(
    "data/android_figurine/test",
    "data/android_figurine/test",
    ["android", "pig_android"],
)

model_file_name = "android_detection.tflite"
save_to_file = "android_evaluated.jpg"
input_image = "android.jpg"
num_threads = 4
detection_threshold = 0.5
epochs = 20
create_model(
    train_data,
    val_data,
    model_file_name=model_file_name,
    epochs=epochs,
    model_name="efficientdet_lite0",
)
detect_on_image(
    model_file_name,
    save_to_file,
    num_threads=num_threads,
    detection_threshold=detection_threshold,
    input_image=input_image,
)
