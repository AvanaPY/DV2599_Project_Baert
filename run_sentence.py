import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import src.tf_versions
import tensorflow as tf
import pandas as pd
import numpy as np
import datetime
import argparse

from src.data import read_data, get_datasets
from src.model import get_modle, train_model, load_model, save_model, compile_model, get_encoder, get_preprocessing_modle
from src.const import mbti_idx2typ, mbti_typ2idx, mbti_p_typs

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                prog = 'run_sentence.py',
                description = 'Tests cool shit',
                epilog = 'This program uses a version of Baert and runs it against a sentence in order to predict the MBTI type of the person who wrote the sentence. It probably sucks.')

    parser.add_argument('--model',      default='beart_2022_12_13__17_39_47', 
                                        help="Which model to use that needs to exist in a /models folder in the root directory.")
    parser.add_argument('--real-flag',  dest='real_flag', 
                                        choices=mbti_p_typs, 
                                        default=None,
                                        help="The real MBTI flag to compare the result to.")
    parser.add_argument('--sentence',   action='store', 
                                        type=str, 
                                        required=True,
                                        help="The sentence at which to run the model against.")
    args = parser.parse_args()

    baert_path = os.path.join('models', args.model)
    if not os.path.exists(baert_path):
        raise RuntimeError(f'Model \"{baert_path}\" does not exist.')
    
    baert = load_model(baert_path)

    data = [ args.sentence ] 

    results = baert.predict(data)
    indices = np.argmax(results, axis=1)
    result = f'{mbti_idx2typ[indices[0]]} ({results[0][indices[0]]:.3f})'

    print(f'Running model {args.model}:')
    print(f'\"{args.sentence}\"\n          -> {result}')

    if args.real_flag:
        print(f'  Real flag: {args.real_flag} ({results[0][mbti_typ2idx[args.real_flag]]:.3f})')

        print('\t  ' + '\t '.join(mbti_p_typs))
        print('\t' + '\t'.join([f'{i:.4f}' for i in results[0]]))