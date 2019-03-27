

import argparse

import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
import keras.backend as K

from keras.models import load_model

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os


def ParseArgs():

    parser = argparse.ArgumentParser()

    parser.add_argument('--model')
    parser.add_argument('--data')
    parser.add_argument('--out_dir')

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


# util function to convert a tensor into a valid image
def deprocessImage(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to array
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    # print(x.shape)
    return x


if __name__ == '__main__':


    args = ParseArgs()


    # tensorflow config allow_growth
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    KTF.set_session(sess)

    x_train_total, y_train_total = training_data_preprocessing(args.data)
    cnn_model = load_model(args.model)

    name_tag = args.model.rsplit('.', 1)[0].split('/')[-1]

    tensor_input = cnn_model.input
    tensor_output = cnn_model.output

    # class 0, 1, 2, 3, 4, 5, 6
    input_img_idx_list = [25704, 25717, 25707, 25705, 25700, 25699, 25718]

    for input_img_idx in input_img_idx_list:

        input_img = x_train_total[input_img_idx]
        pre_label = cnn_model.predict(input_img.reshape(1, 48, 48, 1)).argmax(axis=1)[0]

        print ("Image {} Predicted Label {}".format(input_img_idx, pre_label))

        tensor_target = K.mean(tensor_output[:, pre_label])
        tensor_gradient = K.l2_normalize(K.gradients(tensor_target, tensor_input)[0])
        function_gradient = K.function([tensor_input, K.learning_phase()], [tensor_gradient])

        grad = function_gradient([input_img.reshape(1, 48, 48, 1), 0])[0].reshape(48, 48, 1)
        heatmp = deprocessImage(np.max(np.abs(grad), axis=-1, keepdims=True)).reshape(48, 48) # abs and max of each channel

        # image
        plt.figure()
        plt.imshow(input_img.reshape(48, 48), cmap='gray')
        plt.colorbar()
        plt.tight_layout()
        fig = plt.gcf()
        plt.savefig(os.path.join(args.out_dir, name_tag+".ori_img.image_{}.png".format(input_img_idx)))

        # heatmp
        plt.figure()
        plt.imshow(heatmp, cmap=plt.cm.jet)
        plt.colorbar()
        plt.tight_layout()
        fig = plt.gcf()
        plt.savefig(os.path.join(args.out_dir, name_tag+".heatmp.image_{}.png".format(input_img_idx)))

        # machine_focus
        thres = 0.5 * 255
        machine_focus = input_img.reshape(48, 48)
        machine_focus[np.where(heatmp <= thres)] = np.mean(machine_focus)
        plt.figure()
        plt.imshow(machine_focus, cmap='gray')
        plt.colorbar()
        plt.tight_layout()
        fig = plt.gcf()
        plt.savefig(os.path.join(args.out_dir, name_tag+".machine_focus.image_{}.png".format(input_img_idx)))
