import pandas as pd
import numpy as np
from const import mbti_typ2idx, mbti_typ2idx

def one_hot_encode(v : int, dim : int):
    a = np.zeros(shape=(dim), dtype=np.int32)
    a[v] = 1
    return a

def mbti_to_int(s : pd.Series):
    x = mbti_typ2idx[s["author_flair_text"]]
    # x = one_hot_encode(x, 16)             # Uncomment this line if we use CategoricalCrossentropy()
    return x