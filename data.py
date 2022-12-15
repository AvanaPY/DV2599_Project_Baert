import os
import pandas as pd
import numpy as np
import csv
import tensorflow as tf
from model import BaertModel

from const import mbti_p_typs, MBTI_CLASSES
from preprocessing import mbti_to_int

MAX_ROWS_PER_COUNT = 3000

def apply_mbti_constraint(row : pd.Series):
    for t in mbti_p_typs:
        if t in row["author_flair_text"]:
            row["author_flair_text"] = t
            return

def create_filtered_dataset(fp : str, filtered_fp : str, nrows:int):

    df = pd.read_csv(fp, sep=',', dtype={'author_flair_text': 'str', 
                                            'body'             : 'str',
                                            'subreddit'        : 'str'}, 
                                    nrows=nrows)

    ###############################################################################
    # Filter out all comments that do not have a MBTI type in the author flairs
    # by doing this we assume the people have properly self-diagnosed themselves
    # which they haven't but that's fine, we can work with bad data for the memes
    # man, this sucks
    ###############################################################################
    
    df = df.dropna()
    _filter = df["author_flair_text"] == 0
    for t in mbti_p_typs:
        _f = df["author_flair_text"].str.contains(t)
        _filter |= _f

    df = df[_filter].reset_index(drop=True)

    df["subreddit"] = df["subreddit"].str.upper()

    # Constrain all rows to only contain the MBTI class
    # in the "author_flair_text" column
    df.apply(apply_mbti_constraint, axis=1)

    # Save it to a new file for future use
    df.to_csv(filtered_fp, 
                sep=',', 
                index=False,
                quoting=csv.QUOTE_ALL)
    return df

def read_data(fp : str, filtered_fp : str, nrows=None,limit_row_count:bool=True):
    if nrows == None:
        nrows = 2**25

    if not os.path.exists(filtered_fp):
        df = create_filtered_dataset(fp, filtered_fp, nrows)
    else:
        df = pd.read_csv(filtered_fp, sep=',', 
                            dtype={'author_flair_text': 'str', 
                                   'body'             : 'str',
                                   'subreddit'        : 'str'},
                            nrows=nrows)

    # Add an integer column with integer values that represent the
    # MBTI personality type of each entry which will then be used
    # as our label for the machine learning 
    df["author_flair_text_i"] = df.apply(mbti_to_int, axis=1)
    
    # Remove a bunch of rows in order to balance the
    # dataset so there are a rougly equal amount 
    # of rows for each class
    # also 1.8 million rows is too many 
    if limit_row_count:
        loaded_rows = len(df.index)
        rows_per_count = min(MAX_ROWS_PER_COUNT, int(loaded_rows / MBTI_CLASSES))
        l = [
            df[df['author_flair_text_i'] == i].head(rows_per_count) for i in range(MBTI_CLASSES)
        ]
        return pd.concat(l)
    return df

def get_datasets(data_file : str,
                 filtered_file : str,
                 model : BaertModel,
                 batch_size:int=32, 
                 shuffle_b_size:int=1024,
                 prefetch_size:int=tf.data.AUTOTUNE,
                 nrows:int=2**10,
                 validation_split:float=0.2,
                 limit_row_count:bool=True):

    df = read_data( fp=data_file, 
                    filtered_fp=filtered_file,
                    nrows=nrows,
                    limit_row_count=limit_row_count)
    
    # Turn the data into two numpy arrays, one for data and one for labels
    x = df["body"].to_numpy()
    x = list(x)
    y = np.array(df["author_flair_text_i"].to_numpy())
    y = list(y)

    # Let the model preprocess the raw data
    x, y = model.preprocess_data(x, y)

    # Create a tensorflow DataSet
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if not (shuffle_b_size is None):
        ds = ds.shuffle(buffer_size=shuffle_b_size)
    ds_count = tf.data.experimental.cardinality(ds).numpy()

    # Split it into Training data and Validation data
    train_count = int(ds_count * (1 - validation_split))
    train_ds = ds.take(train_count)
    train_ds = train_ds.batch(batch_size=batch_size)
    train_ds = train_ds.cache().prefetch(buffer_size=prefetch_size)

    val_ds = ds.skip(train_count)
    val_ds = val_ds.batch(batch_size=batch_size)
    val_ds = val_ds.cache().prefetch(buffer_size=prefetch_size)
    return train_ds, val_ds