import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tf_versions
import pandas as pd
import numpy as np
import datetime
from const import mbti_idx2typ

import tensorflow as tf
from data import read_data, get_datasets
from model import get_modle, train_model, load_model, save_model, compile_model

import argparse

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

def restricted_float(x, min_val:float=0.0, max_val:float=1.0):
    try:
        x = float(x)
    except:
        raise argparse.ArgumentTypeError(f'{x} is not a floating-point literal')
    
    if x < min_val or x > max_val:
        raise argparse.ArgumentTypeError(f'{x} is not a floating-point literal between or equal to 0.0 and 1.0')
    return x

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog="train.py",
        description="Trains a Baert NN model"
    )
    parser.add_argument('--epochs', default=10, 
                                            type=int, 
                                            help="How many epochs to run.")
    parser.add_argument('--nrows',      default=2**10,
                                            type=int,
                                            help="How many rows from the data set to load in to train on.")
    parser.add_argument('--val-split',  default=0.1,
                                            type=restricted_float,
                                            help="How large the validation split of data shall be.",
                                            dest='val_split')
    parser.add_argument('--batch-size', default=8,
                                            type=int,
                                            help="How big batches to train on.",
                                            dest='batch_size')
    parser.add_argument('--init-lr',    default=3e-5,
                                            type=restricted_float,
                                            help='Initial learning rate',
                                            dest='init_lr')
    parser.add_argument('--warmup-steps', default=0.1,
                                            type=restricted_float,
                                            help="Fraction of epochs to use as warmup steps for learning rate.",
                                            dest='warmup_frac')
    parser.add_argument('--name',       default=None,
                                            type=str,
                                            help='The name which to give the model')
    parser.add_argument('--add-timestamp', default=True,
                                            type=bool,
                                            help='Whether or not to add a timestamp to the model.',
                                            dest='add_timestamp')
    args = parser.parse_args()

    with tf.device('/gpu:0'):
        # Load the data
        # Create a model
        baert : tf.keras.Model = None
        baert, p_modle, modle = get_modle()

        train_ds, val_ds = get_datasets("data/mbti_full_pull.csv",
                                        "data/mbti_filtered.csv",
                                        baert,
                                        batch_size=args.batch_size,
                                        shuffle_b_size=2**10,
                                        nrows=args.nrows,
                                        validation_split=args.val_split)
        

        # Compile the model against the hyper paramters
        epochs  = args.epochs
        init_lr = args.init_lr
        warmup_frac = args.warmup_frac

        steps_per_epoch = tf.data.experimental.cardinality(train_ds).numpy()
        num_train_steps = steps_per_epoch * epochs
        num_warmup_steps = int(warmup_frac*num_train_steps)

        compile_model(baert, init_lr, num_train_steps, num_warmup_steps)

        if args.name:
            m_name = args.name
        else:
            m_name = "baert" 
            
        if args.add_timestamp:
            m_name += '_' + datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S")

        # Train the model
        train_model(train_ds, val_ds, baert, epochs=epochs, m_name=m_name)

        # Save the model to a file
        model_path = os.path.join("models", m_name)
        save_model(baert, model_path)

        print(f'Saving to \"{model_path}\"')

        print(f'Performance training data set:')
        evaluated = baert.evaluate(train_ds)
        print(f'\tLoss:     {evaluated[0]:.3f}')
        print(f'\tAccuracy: {evaluated[1]:.3f}')
        
        print(f'Performance validation data set:')
        evaluated = baert.evaluate(val_ds)
        print(f'\tLoss:     {evaluated[0]:.3f}')
        print(f'\tAccuracy: {evaluated[1]:.3f}')

        print(f'Saving to \"{model_path}\"')