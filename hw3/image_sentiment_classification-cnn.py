

import argparse
import re

import pandas as pd
import numpy as np

from keras.utils import to_categorical

import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, LeakyReLU, ELU, PReLU, ReLU, BatchNormalization, MaxPooling2D, Dropout, Flatten, Dense
from keras.utils import print_summary, plot_model
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger


DEFAULT_VAL = 0.1
DEFAULT_SEED = None
DEFAULT_EPOCH = 5000
DEFAULT_BATCH = 64
DEFAULT_OPT = 'adam'
DEFAULT_PATIENCE = 100
DEFAULT_FILTERS = 112
DEFAULT_CNN_ACTIVATION = 'leakyrelu'
DEFAULT_CNN_ACTIVATION_ALPHA = 0.3
DEFAULT_UNITS = 896
DEFAULT_DNN_ACTIVATION = 'leakyrelu'
DEFAULT_DNN_ACTIVATION_ALPHA = 0.3

def ParseArgs():

    parser = argparse.ArgumentParser()

    parser.add_argument('--train', help='path of training data; If specified, new model will be trained and output to --out_model.')
    parser.add_argument('--validate', help='float 0~1: the proportions of validation set split from training dataset')
    parser.add_argument('--random_seed', help='random seed for splitting training and validation data')

    parser.add_argument('--epoch', help='number of training epoch')
    parser.add_argument('--batch_size', help='sgd mini batch size')
    parser.add_argument('--optimizer', help='option: \"adam\"')
    parser.add_argument('--filters', help='number of filters in first convolutional layer; the filters will increase by 2 times in preceeding convolutional layer')
    parser.add_argument('--cnn_activation', help='activation function option: \"leakyrelu\", \"elu\", \"prelu\", \"relu\"')
    parser.add_argument('--cnn_activation_alpha', help='alpha for activation function')
    parser.add_argument('--units', help='number of units for dnn input and hidden layers')
    parser.add_argument('--dnn_activation', help='activation function option: \"leakyrelu\", \"elu\", \"prelu\", \"relu\"')
    parser.add_argument('--dnn_activation_alpha', help='alpha for activation function')

    parser.add_argument('--out_log', help='path to output log file')
    parser.add_argument('--out_model', help='path to output log file')

    args = parser.parse_args()


    if args.validate:
        if re.match(r'[0-9]*\.[0-9]*', args.validate):
            args.validate = float(args.validate)
        else:
            raise("Invalid Parameter: {}".format(args.validate))
    else:
        args.validate = DEFAULT_VAL

    if args.random_seed:
        args.random_seed = int(args.random_seed)
    else:
        args.random_seed = DEFAULT_SEED

    if args.epoch:
        args.epoch = int(args.epoch)
    else:
        args.epoch = DEFAULT_EPOCH

    if args.batch_size:
        args.batch_size = int(args.batch_size)
    else:
        args.batch_size = DEFAULT_BATCH

    if args.optimizer:
        if args.optimizer.strip().lower() in ['adam']:
            args.optimizer = args.optimizer.strip().lower()
        else:
            raise("Invalid optimizer: {}".format(args.optimizer))
    else:
        args.optimizer = DEFAULT_OPT


    if args.filters:
        args.filters = int(args.filters)
    else:
        args.filters = DEFAULT_FILTERS

    if args.cnn_activation:
        if args.cnn_activation.strip().lower() in ['leakyrelu', 'elu', 'prelu', 'relu']:
            args.cnn_activation = args.cnn_activation.strip().lower()
        else:
            raise("Invalid activation function: {}".format(args.cnn_activation))
    else:
        args.cnn_activation = DEFAULT_CNN_ACTIVATION

    if args.cnn_activation_alpha:
        args.cnn_activation_alpha = float(args.cnn_activation_alpha)
    else:
        args.cnn_activation_alpha = DEFAULT_CNN_ACTIVATION_ALPHA

    if args.units:
        args.units = int(args.units)
    else:
        args.units = DEFAULT_UNITS

    if args.dnn_activation:
        if args.dnn_activation.strip().lower() in ['leakyrelu', 'elu', 'prelu', 'relu']:
            args.dnn_activation = args.dnn_activation.strip().lower()
        else:
            raise("Invalid activation function: {}".format(args.dnn_activation))
    else:
        args.dnn_activation = DEFAULT_DNN_ACTIVATION

    if args.dnn_activation_alpha:
        args.dnn_activation_alpha = float(args.dnn_activation_alpha)
    else:
        args.dnn_activation_alpha = DEFAULT_DNN_ACTIVATION_ALPHA


    args.args_str = 'val_{}_seed_{}_epoch_{}_batch_{}_opt_{}_cnn_filter_{}_{}_{}_dnn_unit_{}_{}_{}'.format(
        args.validate, args.random_seed, args.epoch, args.batch_size, args.optimizer,
        args.filters, args.cnn_activation, args.cnn_activation_alpha, args.units, args.dnn_activation, args.dnn_activation_alpha
    )


    if not args.out_log:
        args.out_log = './log/{}.log'.format(args.args_str)

    if not args.out_model:
        args.out_model = './model/{}'.format(args.args_str)


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
    y_train_total = to_categorical(raw_train['label'].values, num_classes=7, dtype='int32')

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


    # training data preprocessing and split validation data
    x_train_total, y_train_total = training_data_preprocessing(args.train)
    x_train, y_train, x_val, y_val = split_validate(x_train_total, y_train_total, args.validate, seed=args.random_seed)

    # tensorflow config allow_growth
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    KTF.set_session(sess)


    # cnn model
    cnn_model = Sequential()

    # cnn layers: convolution, activation, batch normalization, max pooling, dropout
    input_shape_list = [ (48, 48, 1), (24, 24, 1), (12, 12, 1), (6, 6, 1) ]
    filters_list = [ (args.filters * 2**i) for i in range(4) ]
    drop_rate_list = [ 0.2, 0.25, 0.3, 0.3 ]

    for layer_i, (input_shape, filters, drop_rate) in enumerate(zip(input_shape_list, filters_list, drop_rate_list)):

        cnn_model.add(
            Conv2D(
                input_shape=input_shape,
                filters=filters,
                kernel_size=(3,3),
                strides=(1, 1),
                padding='same', data_format='channels_last',
                activation=None, use_bias=True,
                kernel_initializer='glorot_uniform', bias_initializer='zeros',
            )
        )

        # LeakyReLU, ELU, PReLU, ReLU
        if args.cnn_activation == 'leakyrelu':
            cnn_model.add( LeakyReLU(alpha=args.cnn_activation_alpha) ) # 0.3
        elif args.cnn_activation == 'elu':
            cnn_model.add( ELU(alpha=args.cnn_activation_alpha) ) # 1.0
        elif args.cnn_activation == 'prelu':
            cnn_model.add( PReLU(alpha_initializer='zeros') )
        elif args.cnn_activation == 'relu':
            cnn_model.add( ReLU(negative_slope=0.0) )

        cnn_model.add( BatchNormalization(axis=-1) )
        cnn_model.add( MaxPooling2D(pool_size=(2, 2), strides=None, padding='same', data_format='channels_last') )
        cnn_model.add( Dropout(rate=drop_rate) )


    # flaten layer
    cnn_model.add(Flatten(data_format='channels_last'))


    # dnn layers: dense, batch normalization, dropout
    # input
    cnn_model.add(
        Dense(
            units=(args.filters * 2**3), activation=None, use_bias=True,
            kernel_initializer='glorot_uniform', bias_initializer='zeros',
        )
    )

    # LeakyReLU, ELU, PReLU, ReLU
    if args.dnn_activation == 'leakyrelu':
        cnn_model.add( LeakyReLU(alpha=args.dnn_activation_alpha) ) # 0.3
    elif args.dnn_activation == 'elu':
        cnn_model.add( ELU(alpha=args.dnn_activation_alpha) ) # 1.0
    elif args.dnn_activation == 'prelu':
        cnn_model.add( PReLU(alpha_initializer='zeros') )
    elif args.dnn_activation == 'relu':
        cnn_model.add( ReLU(negative_slope=0.0) )

    cnn_model.add( BatchNormalization(axis=-1) )
    cnn_model.add( Dropout(rate=0.5) )

    # hidden
    cnn_model.add(
        Dense(
            units=(args.filters * 2**3), activation=None, use_bias=True,
            kernel_initializer='glorot_uniform', bias_initializer='zeros',
        )
    )

    # LeakyReLU, ELU, PReLU, ReLU
    if args.dnn_activation == 'leakyrelu':
        cnn_model.add( LeakyReLU(alpha=args.dnn_activation_alpha) ) # 0.3
    elif args.dnn_activation == 'elu':
        cnn_model.add( ELU(alpha=args.dnn_activation_alpha) ) # 1.0
    elif args.dnn_activation == 'prelu':
        cnn_model.add( PReLU(alpha_initializer='zeros') )
    elif args.dnn_activation == 'relu':
        cnn_model.add( ReLU(negative_slope=0.0) )

    cnn_model.add( BatchNormalization(axis=-1) )
    cnn_model.add( Dropout(rate=0.5) )

    # output
    cnn_model.add(
        Dense(
            units=7, activation='softmax', use_bias=True,
            kernel_initializer='glorot_uniform', bias_initializer='zeros',
        )
    )


    # compile model
    cnn_model.compile( optimizer=args.optimizer, loss='categorical_crossentropy', metrics=['accuracy'] )
    print_summary(cnn_model)


    # callback functions
    check_point = ModelCheckpoint(
        filepath = args.out_model+"_{epoch:04d}-{val_acc:.4f}.hdf5",
        monitor='val_acc',
        verbose=1,
        save_best_only=True, save_weights_only=False,
        mode='auto',
        period=1
    )
    early_stop = EarlyStopping(
        monitor='val_acc',
        min_delta=0,
        patience=DEFAULT_PATIENCE,
        verbose=1,
        mode='auto'
    )
    logger = CSVLogger(args.out_log, separator=',', append=True)


    # image generator
    img_gen = ImageDataGenerator(
        zca_whitening=False, zca_epsilon=1e-06,
        rotation_range=30,
        width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2,
        zoom_range=0.2,
        fill_mode='nearest',
        horizontal_flip=True, vertical_flip=False,
        data_format='channels_last'
    )

    cnn_model.fit_generator(
        generator = img_gen.flow(x_train, y_train, batch_size=args.batch_size),
        steps_per_epoch = np.floor(x_train.shape[0] / args.batch_size),
        epochs=args.epoch,
        verbose=1,
        callbacks=[check_point, logger], # early_stop
        validation_data=(x_val, y_val),
        shuffle=True,
        initial_epoch=0
    )

    cnn_model.save(args.out_model+".hdf5")
