

import argparse

import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

from keras.models import load_model
from keras import backend as K
from keras import layers

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os


DEFAULT_FILTERS = 56
DEFAULT_STEP = 20
DEFAULT_IMG_ID = 2


def ParseArgs():

    parser = argparse.ArgumentParser()

    parser.add_argument('--model')
    parser.add_argument('--data')
    parser.add_argument('--output_dir')

    args = parser.parse_args()


    print ("################## Arguments ##################")

    args_dict = vars(args)
    for key in args_dict:
        print ( "\t{}: {}".format(key, args_dict[key]) )

    print ("###############################################")

    return args


def normalize(x):
    return x / (K.sqrt(K.mean(K.square(x))) + K.epsilon())


def deprocess_image(x):

    # normalize tensor: center on 0., ensure std is 0.25
    x -= x.mean()
    x /= (x.std() + K.epsilon())
    x *= 0.25

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_data_format() == 'channels_first':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def gradient_ascent(filter_activate_img, iterate_func, step_n):

    step_eta = 1

    for step_i in range(step_n):

        activate_value, grad = iterate_func([filter_activate_img, 0])
        filter_activate_img += (step_eta*grad)

        print ("step {}, activate value {}".format(step_i, activate_value))

    return activate_value, filter_activate_img


def plot_activate_filter(model, layer_dict, filter_n, step_n, out_dir, name_tag):

    K.set_image_data_format('channels_last')

    tensor_input = model.inputs[0]
    for layer_name in layer_dict:

        layer = layer_dict[layer_name]

        filter_activate_img_list = []
        for filter_i in range(filter_n):

            print ("#-------------------------------------------------#")
            print ("layer {}, filter {}".format(layer_name, filter_i))
            print ("#-------------------------------------------------#")

            random_noise_img = np.random.random((1, 48, 48, 1)) # init with random noise

            tensor_target = K.mean(layer.output[:, :, :, filter_i])
            tensor_grad = normalize(K.gradients(tensor_target, tensor_input)[0])
            iterate_func = K.function([tensor_input, K.learning_phase()], [tensor_target, tensor_grad])

            activate_value, filter_activate_img = gradient_ascent(random_noise_img, iterate_func, step_n)
            filter_activate_img_list.append((activate_value, filter_activate_img))

        fig = plt.figure(figsize=(14, 17), dpi=250)
        for i, (activate_value, filter_activate_img) in enumerate(filter_activate_img_list):

            ax = fig.add_subplot(filter_n/8, 8, i+1)
            ax.imshow(deprocess_image(filter_activate_img.squeeze()), cmap="Blues")
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))
            plt.xlabel("{:.3f}".format(activate_value))
            plt.tight_layout()

        plt.savefig(os.path.join(out_dir, name_tag+".activate_filter.layer_{}.png".format(layer_name)), dpi=250)


def plot_filtered_img(image, model, layer_dict, filter_n, out_dir, name_tag):

    K.set_image_data_format('channels_last')

    tensor_input = model.inputs[0]
    for layer_name in layer_dict:

        print ("#-------------------------------------------------#")
        print ("layer {}, filter {}".format(layer_name, filter_n))
        print ("#-------------------------------------------------#")

        layer = layer_dict[layer_name]

        layer_output_function = K.function([tensor_input, K.learning_phase()], [layer.output])
        layer_output = layer_output_function([image, 0])[0] # get the output of that layer list (1, 1, 48, 48, 64)

        fig = plt.figure(figsize=(14, 17))
        for filter_i in range(filter_n):
            ax = fig.add_subplot(filter_n/8, 8, filter_i+1)
            ax.imshow(layer_output[0,:,:,filter_i].squeeze(), cmap="Blues")
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))
            plt.xlabel("filter {}".format(filter_i))
            plt.tight_layout()

        plt.savefig(os.path.join(out_dir, name_tag+".filtered_image.layer_{}.png".format(layer_name)), dpi=250)


if __name__ == '__main__':


    args = ParseArgs()


    # tensorflow config allow_growth
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    KTF.set_session(sess)


    cnn_model = load_model(args.model)
    layer_dict = dict( [(layer.name, layer) for layer in cnn_model.layers if ('conv2d' in layer.name) or ('elu' in layer.name)][:8] )

    name_tag = args.model.rsplit('.', 1)[0].split('/')[-1]
    plot_activate_filter(cnn_model, layer_dict, DEFAULT_FILTERS, DEFAULT_STEP, args.output_dir, name_tag)

    raw_data = pd.read_csv(args.data)
    x = np.array([ row.split() for row in raw_data['feature'] ], dtype='float') / 255
    image = x[DEFAULT_IMG_ID].reshape(1, 48, 48, 1)
    plot_filtered_img(image, cnn_model, layer_dict, DEFAULT_FILTERS, args.output_dir, name_tag)
