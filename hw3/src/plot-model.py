

import argparse

from keras.models import load_model
from keras.utils import plot_model

import os


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
        out_path = os.path.join(args.output_dir, out_name)

        print (out_name)

        cnn_model = load_model(file_path)
        cnn_model.summary()
        plot_model(cnn_model, show_shapes=True, to_file=out_path)
