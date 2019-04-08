

import argparse
import re

import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from keras.models import load_model
from keras.utils.generic_utils import get_custom_objects
from keras import backend as K
from keras.layers import Activation

from gensim.models import FastText as ft

import os

import pandas as pd
import numpy as np


DEFAULT_MAX_STR_LEN = 40
DEFAULT_SEED = 1234
DEFAULT_VAL_N = 0.1
DEFAULT_DATA_PROC_KEY = ['raw', 'proc', 'freq_proc', 'stem_frq_pro']


# tensorflow config allow_growth
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
KTF.set_session(sess)

# add swish activation to keras
class Swish(Activation):

    def __init__(self, activation, **kwargs):
        super(Swish, self).__init__(activation, **kwargs)
        self.__name__ = 'swish'

def swish(x):
    return (K.sigmoid(x) * x)

get_custom_objects().update({'swish': Swish(swish)})


def ParseArgs():

    parser = argparse.ArgumentParser()

    parser.add_argument('--w2v')
    parser.add_argument('--data')
    parser.add_argument('--answer')
    parser.add_argument('--validate')
    parser.add_argument('--seed')
    parser.add_argument('--val_thres')
    parser.add_argument('--model_dir')
    parser.add_argument('--output')

    args = parser.parse_args()

    if args.seed:
        args.seed = int(args.seed)
    else:
        args.seed = DEFAULT_SEED


    print ("################## Arguments ##################")

    args_dict = vars(args)
    for key in args_dict:
        print ( "\t{}: {}".format(key, args_dict[key]) )

    print ("###############################################")

    return args


def LoadX(w2v_model, w2v_dim, x_path, str_max_len=40, oov=None):

    with open(x_path, 'r') as in_file:
        x_text_list = [ line.strip() for line in in_file ]

    x = np.zeros(shape=(len(x_text_list), str_max_len, w2v_dim), dtype=float)

    for sen_i, sen in enumerate(x_text_list):
        for word_i, word in enumerate(sen.split()):

            if not (word in w2v_model.wv.vocab) and not (oov is None):
                x[sen_i][word_i] = w2v_model.wv.word_vec(oov)
            else:
                x[sen_i][word_i] = w2v_model.wv.word_vec(word)

    return x


def LoadY(y_path):

    with open(y_path, 'r') as in_file:
        y = np.array([ float(y.strip()) for y in in_file], dtype=int)

    return y


def SplitValidate(train_x, train_y, validate_n, seed=None):

    # val_n
    data_n = train_x.shape[0]
    if isinstance(validate_n, float):
        val_n = int(data_n*validate_n)
    elif isinstance(validate_n, int):
        val_n = int(validate_n)

    # mask
    mask = np.ones(data_n, dtype=bool)

    sample_idx = np.arange(data_n)
    if seed:
        rng = np.random.RandomState(int(seed))
        rng.shuffle(sample_idx)
    else:
        np.random.shuffle(sample_idx)
    sample_idx = sample_idx[:val_n]

    mask[sample_idx] = False

    # train, validate
    x_train, y_train = train_x[mask], train_y[mask]
    x_val, y_val = train_x[~mask], train_y[~mask]

    print ("Training data x shape: {}, Training ground truth y shape: {}".format(x_train.shape, y_train.shape))
    print ("Validation data x shape: {}, Validation ground truth y shape: {}".format(x_val.shape, y_val.shape))

    return x_train, y_train, x_val, y_val


if __name__ == '__main__':


    args = ParseArgs()

    # load data
    x_dict = {}
    for data_proc_key in DEFAULT_DATA_PROC_KEY:

        # w2v model
        ft_in_path = "{}.{}.bin".format(args.w2v, data_proc_key)
        ft_model = ft.load_fasttext_format(ft_in_path)

        # x
        x_in_path = "{}.{}.txt".format(args.data, data_proc_key)
        x = LoadX(ft_model, ft_model.vector_size, x_in_path, str_max_len=DEFAULT_MAX_STR_LEN, oov=None)
        if args.answer:
            y = LoadY(args.answer)
        if args.validate:
            _, _, x, y = SplitValidate(x, y, float(args.validate), seed=args.seed)
        x_dict[data_proc_key] = x

    # model path
    file_path_list = [ os.path.join(args.model_dir, f) for f in os.listdir(args.model_dir) ]

    # model predict
    prediction = []
    for path in file_path_list:

        if args.val_thres:
            val_acc = float( re.search(r".*_val_acc_(\d*.\d*)_.*", path).group(1) )
            if val_acc < float(args.val_thres):
                continue

        print ("Using model: {}".format(path))

        for data_proc_key in DEFAULT_DATA_PROC_KEY:
            if data_proc_key in path:
                x = x_dict[data_proc_key]
                break
        rnn_model = load_model(path)
        prediction.append(rnn_model.predict(x).argmax(axis=1))

    prediction = np.array(prediction)

    # ensemble
    ensemble_ans = [ np.bincount(prediction[:, idx]).argmax() for idx in range(x.shape[0]) ]

    # accuracy
    if args.answer:

        acc = 0.0
        for idx in range(y.shape[0]):
            if (np.bincount(prediction[:, idx]).argmax() == y[idx]):
                acc += 1
        acc /= y.shape[0]

        print ("Accuracy {:.4f}".format(acc))

    # output ensemble prediction
    if args.output:

        with open(args.output, 'w') as out_file:

            out_file.write('id,label\n')

            for idx, ans in enumerate(ensemble_ans):
                out_file.write('{},{}\n'.format(idx, ans))
