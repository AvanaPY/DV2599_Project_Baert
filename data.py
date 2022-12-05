import os
import pandas as pd
import numpy as np
import csv
import tensorflow as tf

from const import mbti_p_typs
from preprocessing import mbti_to_int

def apply_mbti_constraint(row : pd.Series):
    for t in mbti_p_typs:
        if t in row["author_flair_text"]:
            row["author_flair_text"] = t
            return

def read_data(fp : str, filtered_fp : str, nrows=2**10):
    if not os.path.exists(filtered_fp):
        df = pd.read_csv(fp, sep=',', dtype={'author_flair_text': 'str', 
                                             'body'             : 'str',
                                             'subreddit'        : 'str'}, 
                                        nrows=nrows)

        ###############################################################################
        # Filter out all comments that do not have a MBTI type in the author flairs
        # by doing this we assume the people have properly self-diagnosed themselves
        # which they haven't but that's fine, we can work with bad data for the memes
        # man, this sucks
        df = df.dropna()
        _filter = df["author_flair_text"] == 0
        for t in mbti_p_typs:
            _f = df["author_flair_text"].str.contains(t)
            _filter |= _f
        df = df[_filter].set_index(keys=np.arange(len(df[_filter].index)))
        ############################################################################
        df["subreddit"] = df["subreddit"].str.upper()
        df.apply(apply_mbti_constraint, axis=1)
        df.to_csv(filtered_fp, 
                    sep=',', 
                    index=False,
                    quoting=csv.QUOTE_ALL)
    else:
        df = pd.read_csv(filtered_fp, sep=',', 
                            dtype={'author_flair_text': 'str', 
                                   'body'             : 'str',
                                   'subreddit'        : 'str'},
                            nrows=nrows)

    df["author_flair_text_i"] = df.apply(mbti_to_int, axis=1)
    return df

def get_datasets(batch_size:int=32, 
                 shuffle_b_size:int=1024,
                 prefetch_size:int=tf.data.AUTOTUNE,
                 nrows:int=2**15,
                 validation_split:float=0.2):
    df = read_data("mbti_full_pull.csv", 
                    "mbti_filtered.csv",
                    nrows=nrows)
    x = df["body"].to_numpy()
    x = list(x)
    y = np.array(df["author_flair_text_i"].to_numpy())
    y = list(y)

    print(f'x types: {set([type(a) for a in x])}')
    print(f'y types: {set([type(a) for a in y])}')

    ds = tf.data.Dataset.from_tensor_slices((x, y))
    ds = ds.shuffle(buffer_size=shuffle_b_size)
    ds_count = tf.data.experimental.cardinality(ds).numpy()

    train_count = int(ds_count * (1 - validation_split))
    train_ds = ds.take(train_count)
    train_ds = train_ds.batch(batch_size=batch_size)
    train_ds = train_ds.cache().prefetch(buffer_size=prefetch_size)

    val_ds = ds.skip(train_count)
    val_ds = val_ds.batch(batch_size=batch_size)
    val_ds = val_ds.cache().prefetch(buffer_size=prefetch_size)
    
    print(df.head(5))
    print(f'Available flairs: {df["author_flair_text"].unique()}')
    print(f'Size of data: {ds_count} samples')
    return train_ds, val_ds