

import argparse

import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from keras.models import load_model

import os

import pandas as pd
import numpy as np


DEFAULT_SEED = 1234
DEFAULT_VAL_N = 0.1


def ParseArgs():

    parser = argparse.ArgumentParser()

    parser.add_argument('--data')
    parser.add_argument('--answer')
    parser.add_argument('--val')
    parser.add_argument('--model_dir')
    parser.add_argument('--output')

    args = parser.parse_args()


    print ("################## Arguments ##################")

    args_dict = vars(args)
    for key in args_dict:
        print ( "\t{}: {}".format(key, args_dict[key]) )

    print ("###############################################")

    return args


if __name__ == '__main__':


    args = ParseArgs()


    # tensorflow config allow_growth
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    KTF.set_session(sess)

    # read data
    raw_data = pd.read_csv(args.data)
    x = np.array([ row.split() for row in raw_data['feature'] ], dtype='float') / 255
    x = x.reshape( (x.shape[0], 48, 48, 1) )

    if args.val.lower() in ['yes', 'y', 'true', 't']:

        data_n = x.shape[0]
        val_n = int(data_n* DEFAULT_VAL_N)
        mask = np.ones(data_n, dtype=bool)

        sample_idx = np.arange(data_n)
        rng = np.random.RandomState(DEFAULT_SEED)
        rng.shuffle(sample_idx)
        sample_idx = sample_idx[:val_n]
        mask[sample_idx] = False

        x = x[~mask]

    # model path
    file_path_list = [ os.path.join(args.model_dir, f) for f in os.listdir(args.model_dir) ]

    # model predict
    prediction = []
    for path in file_path_list:

        print ("Using model: {}".format(path))

        cnn_model = load_model(path)
        prediction.append(cnn_model.predict(x).argmax(axis=1))

    prediction = np.array(prediction)

    # ensemble
    ensemble_ans = [ np.bincount(prediction[:, idx]).argmax() for idx in range(x.shape[0]) ]

    # accuracy
    if args.answer.lower() in ['yes', 'y', 'true', 't']:

        raw_data = pd.read_csv(args.data)
        answer = raw_data['label'].values

        if args.val.lower() in ['yes', 'y', 'true', 't']:
            answer = answer[~mask]

        acc = 0.0
        for idx in range(answer.shape[0]):
            if (np.bincount(prediction[:, idx]).argmax() == answer[idx]):
                acc += 1
        acc /= answer.shape[0]

        print ("Accuracy {:.4f}".format(acc))

    # output ensemble prediction
    if args.output:

        with open(args.output, 'w') as out_file:

            out_file.write('id,label\n')

            for idx, ans in enumerate(ensemble_ans):
                out_file.write('{},{}\n'.format(idx, ans))
