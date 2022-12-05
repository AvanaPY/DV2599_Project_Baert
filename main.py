import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import pandas as pd
import numpy as np
import tensorflow as tf
from math import ceil
from const import mbti_p_typs
from data import read_data
from model import get_modle, train_model
from preprocessing import apply_bert, mbti_to_int

physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs:", len(physical_devices))

df = read_data('mbti_full_pull.csv', 'mbti_filtered.csv')

x = df["body"].tolist()
df["author_flair_text_i"] = df.apply(mbti_to_int, axis=1)
y = df["author_flair_text_i"].tolist()

train_ds = tf.data.Dataset.from_tensor_slices((x, y))
train_ds = train_ds.shuffle(buffer_size=128).batch(32)
train_ds = train_ds.cache().prefetch(buffer_size=64)

print(df.head(5))
print(f'Available flairs: {df["author_flair_text"].unique()}')

baert, p_modle, modle = get_modle()

train_model(train_ds, baert, epochs=5)