import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import pandas as pd
import numpy as np
import tensorflow as tf
from data import read_data, get_datasets
from model import get_modle, train_model

print(f'Running Tensorflow version {tf.__version__}')

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only use the first GPU
    try:
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=2**12)])

        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)
else:
    print('GPU acceleration available, exiting')
    exit(0)

with tf.device('/gpu:0'):
    train_ds, val_ds = get_datasets(batch_size=16,
                                    shuffle_b_size=2**12,
                                    nrows=2**15,
                                    validation_split=0.2)
    baert, p_modle, modle = get_modle()

    train_model(train_ds, val_ds, baert, epochs=100)