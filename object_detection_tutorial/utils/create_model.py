import numpy as np

from tflite_model_maker import model_spec
from tflite_model_maker import object_detector

import tensorflow as tf
assert tf.__version__.startswith('2')


def create_model(train_data, val_data, batch_size=4, epochs=20, model_name='efficientdet_lite0', model_file_name='dog_detection.tflite'):
    print('Starting')
    print(f"Getting model ${model_name}")
    spec = model_spec.get(model_name)
    print(f"Creating model")
    model = object_detector.create(train_data, model_spec=spec, batch_size=batch_size, train_whole_model=True, epochs=epochs, validation_data=val_data)
    print(f"Evaluating model")
    model.evaluate(val_data)
    print(f"Exporting model to ${model_file_name}")
    model.export(export_dir='.', tflite_filename=model_file_name)
    print(f"Finall evaluation of model to ${model_file_name}")
    model.evaluate_tflite(model_file_name, val_data)
    print('Ending')
