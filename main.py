import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import pandas as pd
import numpy as np
import tensorflow as tf
from data import read_data, get_datasets
from model import get_modle, train_model

physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs:", len(physical_devices))

train_ds, val_ds = get_datasets()

baert, p_modle, modle = get_modle()

train_model(train_ds, baert, epochs=5)