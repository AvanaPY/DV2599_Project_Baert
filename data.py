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

def read_data(fp : str, filtered_fp : str):
    if not os.path.exists(filtered_fp):
        df = pd.read_csv(fp, sep=',', dtype={'author_flair_text': 'str', 
                                             'body'             : 'str',
                                             'subreddit'        : 'str'})

        ###############################################################################
        # Filter out all comments that do not have a MBTI type in the author flairs
        # by doing this we assume the people have properly self-diagnosed themselves
        # which they haven't but that's fine, we can work with bad data for the memes
        # man, this sucks
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
                            nrows=2**14)

    return df

def get_datasets():
    BATCH_SIZE = 64
    BUF_SIZE   = 128

    df = read_data("mbti_full_pull.csv", "mbti_filtered.csv")
    x = df["body"].tolist()
    df["author_flair_text_i"] = df.apply(mbti_to_int, axis=1)
    y = df["author_flair_text_i"].tolist()

    train_ds = tf.data.Dataset.from_tensor_slices((x, y))
    train_ds = train_ds.shuffle(buffer_size=BUF_SIZE).batch(BATCH_SIZE)
    train_ds = train_ds.cache().prefetch(buffer_size=BUF_SIZE)

    val_ds = None
    
    print(df.head(5))
    print(f'Available flairs: {df["author_flair_text"].unique()}')
    return train_ds, val_ds