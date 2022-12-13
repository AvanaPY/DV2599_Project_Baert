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
    print('GPU acceleration unavailable, exiting')
    exit(0)

def create_and_train():
    with tf.device('/gpu:0'):
        # Load the data
        train_ds, val_ds = get_datasets("mbti_full_pull.csv",
                                        "mbti_filtered.csv",
                                        batch_size=8,
                                        shuffle_b_size=2**10,
                                        nrows=2**12,
                                        validation_split=0.2)
        
        # Create a model
        baert : tf.keras.Model = None
        baert, p_modle, modle = get_modle()

        # Compile the model against the hyper paramters
        epochs = 1000
        init_lr = 3e-5
        steps_per_epoch = tf.data.experimental.cardinality(train_ds).numpy()
        num_train_steps = steps_per_epoch * epochs
        num_warmup_steps = int(0.1*num_train_steps)

        compile_model(baert, init_lr, num_train_steps, num_warmup_steps)

        # Train the model
        m_name = datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
        train_model(train_ds, val_ds, baert, epochs=epochs, m_name=m_name)

        # Save the model to a file
        model_name = "beart_" + m_name
        model_path = os.path.join("models", model_name)
        save_model(baert, model_path)

def evaluate():
    baert = load_model(os.path.join('models', 'beart_2022_12_13__17_39_47'))

    train_ds, _ = get_datasets("mbti_full_pull.csv",
                                "mbti_filtered.csv",
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

    loss, acc = baert.evaluate(train_ds)
    print(f'Accuracy: {acc:.3f}\nLoss:     {loss:.3f}\n')

    sntz = [
        "I like cheese.", 
        "I don't like INTP people.", 
        "I really fucking dislike INTJ",
        "Suck my cock.",
        "ESTP people are kind of cringe, not gonna lie.",
        "You should :) it will help you if you just don't suck at video games."
    ]
    t = baert.predict(sntz)
    classes = np.argmax(t, axis=1)
    for i, (snt, c) in enumerate(zip(sntz, classes)):
        print(f'{mbti_idx2typ[c]} ({c:2d}, {t[i][c]:.5f}) <-- {snt}')

def train():
    create_and_train()

if __name__ == '__main__':
    evaluate()
    # train()