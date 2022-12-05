import pandas as pd
import numpy as np
from const import mbti_p_typs_d, mbti_p_typs_d
# How many items per iteration we shall
# give BERT because BERT is old and cannot
# chop many trees at once
# so we need to give him a little bit less job
# per iteration
# but then we can just force him to work
# multiple iterations so he still does all the work
#ModernSlavery2022
MAX_PER_ITER = 1_000

def one_hot_encode(v : int, dim : int):
    a = np.zeros(shape=(dim), dtype=np.int32)
    a[v] = 1
    return a

def mbti_to_int(s : pd.Series):
    x = mbti_p_typs_d[s["author_flair_text"]]
    x = one_hot_encode(x, 16)
    return x

def apply_bert(df : pd.DataFrame, p_model, model):
    bodies = df["body"].tolist();

    pooled   = []
    for i in range(0, len(bodies), MAX_PER_ITER):
        body = bodies[i:i + MAX_PER_ITER]
        bodies_preproc = p_model(body)
        bodies_mathed  = model(bodies_preproc)

        poolled = bodies_mathed["pooled_output"].numpy()
        pooled = pooled + [tuple(a) for a in poolled]

    df["pooled"] = pd.Series(pooled)