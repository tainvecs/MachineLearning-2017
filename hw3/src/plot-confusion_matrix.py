

import argparse

import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

from keras.models import load_model
from sklearn.metrics import confusion_matrix as confu_matrix

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DEFAULT_SEED = 1234
DEFAULT_VAL_N = 0.1


def ParseArgs():

    parser = argparse.ArgumentParser()

    parser.add_argument('--model')
    parser.add_argument('--data')
    parser.add_argument('--output')

    args = parser.parse_args()


    print ("################## Arguments ##################")

    args_dict = vars(args)
    for key in args_dict:
        print ( "\t{}: {}".format(key, args_dict[key]) )

    print ("###############################################")

    return args


def training_data_preprocessing(in_path):

    raw_train = pd.read_csv(in_path)

    # normalize x by 255, encode y category with one-hot encoding
    x_train_total = np.array([ row.split() for row in raw_train['feature'] ], dtype='float') / 255
    y_train_total = raw_train['label'].values

    # reshape data format channels_last for cnn model
    x_train_total = x_train_total.reshape( (x_train_total.shape[0], 48, 48, 1) )

    return x_train_total, y_train_total


def split_validate(train_x, train_y, validate_n, seed=None):

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


    # tensorflow config allow_growth
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    KTF.set_session(sess)


    x_train_total, y_train_total = training_data_preprocessing(args.data)
    x_train, y_train, x_val, y_val = split_validate(x_train_total, y_train_total, DEFAULT_VAL_N, seed=DEFAULT_SEED)

    cnn_model = load_model(args.model)
    prediction_val = cnn_model.predict(x_val).argmax(axis=1)

    class_list = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
    confusion_matrix = confu_matrix(y_val, prediction_val).astype("float")
    confusion_matrix = confusion_matrix / confusion_matrix.sum(axis=1)[:, np.newaxis]

    fig_pixel_x, fig_pixel_y = 1800, 1580
    fig_dpi = 200
    plt.figure(figsize=(fig_pixel_x/fig_dpi, fig_pixel_y/fig_dpi), dpi=fig_dpi)

    plt.imshow(confusion_matrix, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xticks(np.arange(len(class_list)), class_list, rotation=30)
    plt.yticks(np.arange(len(class_list)), class_list)

    thresh = confusion_matrix.max() / 2.0
    for row_i in range(confusion_matrix.shape[0]):
        for col_j in range(confusion_matrix.shape[1]):
            plt.text(
                col_j, row_i, '{:.3f}'.format(confusion_matrix[row_i, col_j]),
                horizontalalignment="center",
                color="white" if confusion_matrix[row_i, col_j] > thresh else "black"
            )

    # plt.tight_layout()
    plt.xlabel("Predicted label")
    plt.ylabel("True label")

    plt.savefig(args.output, dpi=250, transparent=False)
