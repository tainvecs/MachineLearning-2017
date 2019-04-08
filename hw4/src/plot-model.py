

import argparse

from keras.models import load_model
from keras.utils import plot_model

import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

from keras.utils.generic_utils import get_custom_objects
from keras import backend as K

from keras.layers import Activation

import os


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

    parser.add_argument('--model_dir')
    parser.add_argument('--output_dir')

    args = parser.parse_args()


    print ("################## Arguments ##################")

    args_dict = vars(args)
    for key in args_dict:
        print ( "\t{}: {}".format(key, args_dict[key]) )

    print ("###############################################")

    return args


if __name__ == '__main__':


    args = ParseArgs()


    for file_name in os.listdir(args.model_dir):

        file_path = os.path.join(args.model_dir, file_name)
        out_name = "{}.model.png".format(file_name.rsplit('.', 1)[0])
        if '_thread_16.x_train.all' in out_name:
            out_name = out_name.replace('_thread_16.x_train.all', '')
        out_path = os.path.join(args.output_dir, out_name)

        print (out_name)

        rnn_model = load_model(file_path)
        rnn_model.summary()
        plot_model(rnn_model, show_shapes=True, to_file=out_path)
