

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
    parser.add_argument('--val_thres')
    parser.add_argument('--cfd_upper_bound')
    parser.add_argument('--cfd_lower_bound')
    parser.add_argument('--model_dir')
    parser.add_argument('--in_x_prob')
    parser.add_argument('--out_x')
    parser.add_argument('--out_y')
    parser.add_argument('--out_x_prob')

    args = parser.parse_args()

    print ("################## Arguments ##################")

    args_dict = vars(args)
    for key in args_dict:
        print ( "\t{}: {}".format(key, args_dict[key]) )

    print ("###############################################")

    return args


def LoadX(w2v_model, w2v_dim, x_path, load_range, str_max_len=40, oov=None):

    with open(x_path, 'r') as in_file:
        x_text_list = [ line.strip() for line in in_file if line.strip()!='' ][load_range[0]:load_range[1]]

    x = np.zeros(shape=(len(x_text_list), str_max_len, w2v_dim), dtype=float)

    for sen_i, sen in enumerate(x_text_list):
        for word_i, word in enumerate(sen.split()):

            if word_i >= str_max_len:
                continue

            if not (word in w2v_model.wv.vocab) and not (oov is None):
                x[sen_i][word_i] = w2v_model.wv.word_vec(oov)
            else:
                x[sen_i][word_i] = w2v_model.wv.word_vec(word)

    return x


if __name__ == '__main__':


    args = ParseArgs()


    # get data n
    with open("{}.{}.txt".format(args.data, 'raw'), 'r') as in_file:
        data_n = len([ line for line in in_file if line.strip()!='' ])


    new_x_i, new_y = [], []
    if not args.in_x_prob:


        # w2v model
        print ("Loading word2vec model...")
        w2v_dict = {}
        for data_proc_key in DEFAULT_DATA_PROC_KEY:
            ft_in_path = "{}.{}.bin".format(args.w2v, data_proc_key)
            w2v_dict[data_proc_key] = ft.load_fasttext_format(ft_in_path)


        # rnn model
        print ("Loading RNN model...")
        rnn_model_dict = {}
        file_path_list = [ os.path.join(args.model_dir, f) for f in os.listdir(args.model_dir) ]
        for path in file_path_list:

            if args.val_thres:
                val_acc = float( re.search(r".*_val_acc_(\d*.\d*)_.*", path).group(1) )
                if val_acc < float(args.val_thres):
                    continue
            rnn_model_dict[path] = load_model(path)


        slice_n = 200000
        all_predictions = np.zeros((data_n, 2), dtype='float')
        for i in range( int(np.ceil(data_n/slice_n)) ):

            print ("Loading x slice {} / {}".format(i+1, int(np.ceil(data_n/slice_n))))

            # load x
            x_dict = {}
            load_range = [ i*slice_n, min((i+1)*slice_n, data_n) ]
            for data_proc_key in DEFAULT_DATA_PROC_KEY:
                ft_model = w2v_dict[data_proc_key]
                x_in_path = "{}.{}.txt".format(args.data, data_proc_key)
                x = LoadX(ft_model, ft_model.vector_size, x_in_path, load_range, str_max_len=DEFAULT_MAX_STR_LEN, oov=None)
                x_dict[data_proc_key] = x

            # model predict
            prediction_prob = np.zeros((x.shape[0], 2), dtype='float')
            for path in rnn_model_dict:

                print ("Using model: {}".format(path))

                for data_proc_key in DEFAULT_DATA_PROC_KEY:
                    if data_proc_key in path:
                        x = x_dict[data_proc_key]
                        break

                rnn_model = rnn_model_dict[path]
                prediction_prob += rnn_model.predict(x)

            prediction_prob /= len(rnn_model_dict)
            all_predictions[i*slice_n:i*slice_n+prediction_prob.shape[0]] = prediction_prob

            # new data
            for j in range(prediction_prob.shape[0]):

                # rules
                if float(args.cfd_lower_bound) <= (prediction_prob[j][0] - prediction_prob[j][1]) <= float(args.cfd_upper_bound):
                    new_y.append('0')
                    new_x_i.append(i*slice_n + j)
                elif -float(args.cfd_upper_bound) <= (prediction_prob[j][0] - prediction_prob[j][1]) <= -float(args.cfd_lower_bound):
                    new_y.append('1')
                    new_x_i.append(i*slice_n + j)
                else:
                    continue

    else:

        all_predictions = np.load(args.in_x_prob)

        # new data
        for i in range(all_predictions.shape[0]):

            # rules
            if float(args.cfd_lower_bound) <= (all_predictions[i][0] - all_predictions[i][1]) <= float(args.cfd_upper_bound):
                new_y.append('0')
                new_x_i.append(i)
            elif -float(args.cfd_upper_bound) <= (all_predictions[i][0] - all_predictions[i][1]) <= -float(args.cfd_lower_bound):
                new_y.append('1')
                new_x_i.append(i)
            else:
                continue


    # output
    for data_proc_key in DEFAULT_DATA_PROC_KEY:

        with open("{}.{}.txt".format(args.data, data_proc_key), 'r') as in_file:
            x_list = [ line.strip() for line in in_file if line.strip()!='' ]

        with open("{}.{}.txt".format(args.out_x, data_proc_key), 'w') as out_file:
            for i in new_x_i:
                out_file.write("{}\n".format(x_list[i]))

    with open(args.out_y, 'w') as out_file:
        out_file.write('\n'.join(new_y))

    if not args.in_x_prob:
        np.save(args.out_x_prob, all_predictions)
