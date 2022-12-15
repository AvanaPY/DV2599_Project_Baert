import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import pandas as pd
import numpy as np
import tensorflow as tf
import datetime
from const import mbti_idx2typ
from data import read_data, get_datasets
from model import get_modle, train_model, load_model, save_model, compile_model

print(f'Running Tensorflow version {tf.__version__}')

if __name__ == '__main__':
    baert = load_model(os.path.join('models', 'beart_2022_12_13__17_39_47'))

    train_ds, _ = get_datasets( "data/mbti_full_pull.csv",
                                "data/mbti_filtered.csv",
                                batch_size=32,
                                shuffle_b_size=2**10,
                                nrows=2**12,
                                validation_split=0,
                                limit_row_count=False)

    epochs = 1000
    init_lr = 3e-5
    steps_per_epoch = tf.data.experimental.cardinality(train_ds).numpy()
    num_train_steps = steps_per_epoch * epochs
    num_warmup_steps = int(0.1*num_train_steps)
    compile_model(baert, init_lr, num_train_steps, num_warmup_steps)
    baert.summary()

    with tf.device('cpu'):

        # loss, acc = baert.evaluate(train_ds)
        # print(f'Accuracy: {acc:.3f}\nLoss:     {loss:.3f}\n')

        sntz = [
            "I like cheese.", 
            "I don't like INTP people.", 
            "I really fucking dislike INTJ",
            "Suck my cock.",
            "ESTP people are kind of cringe, not gonna lie.",
            "You should :) it will help you if you just don't suck cock."
        ]
        t = baert.predict(sntz)
        classes = np.argmax(t, axis=1)
        for i, (snt, c) in enumerate(zip(sntz, classes)):
            print(f'{mbti_idx2typ[c]} ({c:2d}, {t[i][c]:.3f}) <-- {snt}')