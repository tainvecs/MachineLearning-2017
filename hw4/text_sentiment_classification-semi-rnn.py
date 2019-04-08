

import argparse
import re

import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

from keras.utils.generic_utils import get_custom_objects
from keras import backend as K

from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Bidirectional, RNN, LSTM, GRU, Dense, BatchNormalization, Dropout, Activation
from keras.utils import print_summary, plot_model
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger

from gensim.models import FastText as ft
import numpy as np


DEFAULT_MODEL = 'gru'
DEFAULT_EPOCH = 30
DEFAULT_VAL = 0.1
DEFAULT_VAL_SEED = 1234
DEFAULT_PATIENCE = 100

DEFAULT_MAX_STR_LEN = 40
DEFAULT_W2V_DIM = 256

DEFAULT_RNN_ACTIVATION = 'tanh'
DEFAULT_RNN_UNIT = [256, 256]
DEFAULT_RNN_DROPOUT = [0.4, 0.4]
DEFAULT_RNN_R_DROPOUT = [0.4, 0.4]
DEFAULT_RNN_IN_SHAPE = [ (DEFAULT_MAX_STR_LEN, DEFAULT_W2V_DIM) ] + [ (DEFAULT_MAX_STR_LEN, u_size*2) for u_size in DEFAULT_RNN_UNIT[:-1] ]
DEFAULT_RNN_RT_SEQ = [True]*(len(DEFAULT_RNN_UNIT)-1) + [False]

DEFAULT_DNN_ACTIVATION = 'selu'
DEFAULT_DNN_UNIT = [256, 256]
DEFAULT_DNN_BATCH_NORM = False
DEFAULT_DNN_DROPOUT = [0.3, 0.3]


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

    parser.add_argument('--x_train')
    parser.add_argument('--y_train')
    parser.add_argument('--validate')
    parser.add_argument('--random_seed')
    parser.add_argument('--w2v')
    parser.add_argument('--x_semi')
    parser.add_argument('--y_semi')

    parser.add_argument('--model')
    parser.add_argument('--epoch')
    parser.add_argument('--batch_size')
    parser.add_argument('--optimizer')
    parser.add_argument('--rnn_activation')
    parser.add_argument('--dnn_activation')
    parser.add_argument('--dnn_norm')

    parser.add_argument('--out_log')
    parser.add_argument('--out_model')

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
        args.random_seed = DEFAULT_VAL_SEED

    if args.model:
        if args.model.strip().lower() in ['gru', 'lstm']:
            args.model = args.model.strip().lower()
        else:
            raise("Invalid RNN Model Option: {}".format(args.model))
    else:
        args.model = DEFAULT_MODEL

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
            raise("Invalid Optimizer: {}".format(args.optimizer))
    else:
        args.optimizer = DEFAULT_OPT

    if args.rnn_activation:
        if args.rnn_activation.strip().lower() in ['tanh']:
            args.rnn_activation = args.rnn_activation.strip().lower()
        else:
            raise("Invalid Activation Function: {}".format(args.rnn_activation))
    else:
        args.rnn_activation = DEFAULT_RNN_ACTIVATION

    if args.dnn_activation:
        if args.dnn_activation.strip().lower() in ['selu', 'swish']:
            args.dnn_activation = args.dnn_activation.strip().lower()
        else:
            raise("Invalid Activation Function: {}".format(args.dnn_activation))
    else:
        args.dnn_activation = DEFAULT_DNN_ACTIVATION

    if args.dnn_norm:
        if args.dnn_norm.strip().lower() in ['t', 'true', 'yes', 'y']:
            args.dnn_norm = True
        else:
            args.dnn_norm = False
    else:
        args.dnn_norm = DEFAULT_DNN_BATCH_NORM

    w2v_str = args.w2v.split('/')[-1]
    args.args_str = 'val_{}_seed_{}_epoch_{}_batch_{}_opt_{}_rnn_{}_{}_u_{}_d_{}_rd_{}_dnn_{}_u_{}_d_{}_norm_{}_w2v_{}'.format(
        args.validate, args.random_seed,
        args.epoch, args.batch_size, args.optimizer,
        args.model, args.rnn_activation, '_'.join([str(r) for r in DEFAULT_RNN_UNIT]), DEFAULT_RNN_DROPOUT[0], DEFAULT_RNN_R_DROPOUT[0],
        args.dnn_activation, '_'.join([str(r) for r in DEFAULT_DNN_UNIT]), DEFAULT_DNN_DROPOUT[0], args.dnn_norm,
        w2v_str
    )

    if not args.out_log:
        args.out_log = './log-semi/{}.log'.format(args.args_str)

    if not args.out_model:
        args.out_model = './model-semi/{}'.format(args.args_str)


    print ("################## Arguments ##################")

    args_dict = vars(args)
    for key in args_dict:
        print ( "\t{}: {}".format(key, args_dict[key]) )

    print ("###############################################")

    return args


def LoadXTrain(w2v_model, w2v_dim, x_train_path, str_max_len=40, oov=None):

    with open(x_train_path, 'r') as in_file:
        train_text_list = [ line.strip() for line in in_file ]

    x_train = np.zeros(shape=(len(train_text_list), str_max_len, w2v_dim), dtype=float)

    for sen_i, sen in enumerate(train_text_list):
        for word_i, word in enumerate(sen.split()):

            if word_i >= 40: continue

            if not (word in w2v_model.wv.vocab) and not (oov is None):
                x_train[sen_i][word_i] = w2v_model.wv.word_vec(oov)
            else:
                x_train[sen_i][word_i] = w2v_model.wv.word_vec(word)

    return x_train


def LoadYTrain(y_train_path):

    with open(y_train_path, 'r') as in_file:
        y_train = np.array([ float(y.strip()) for y in in_file ], dtype=int)

    return to_categorical(y_train, num_classes=2, dtype='int32')


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

    return x_train, y_train, x_val, y_val


def ConcatSemiTrain(w2v_model, w2v_dim, x_semi_path, y_semi_path, str_max_len, oov, x_train, y_train):

    #x_semi = LoadXTrain(w2v_model, w2v_dim, x_semi_path, str_max_len, oov)
    #y_semi = LoadYTrain(y_semi_path)
    return np.concatenate((x_train, LoadXTrain(w2v_model, w2v_dim, x_semi_path, str_max_len, oov)), axis=0), np.concatenate((y_train, LoadYTrain(y_semi_path)), axis=0)


def DefineRNN(model, rnn_unit_list, rnn_in_shape_list, rnn_activation, rnn_dropout_list, rnn_r_dropout_list, rnn_rt_seq,
                dnn_unit_list, dnn_activation, dnn_batch_norm, dnn_dropout_list):

    rnn_model = Sequential()

    # bi-directional rnn
    for i in range(len(rnn_unit_list)):

        if model == 'gru':
            rnn_model.add(
                Bidirectional(
                    GRU(
                        input_shape=rnn_in_shape_list[i],
                        units=rnn_unit_list[i],
                        activation=rnn_activation, recurrent_activation='hard_sigmoid',
                        use_bias=True,
                        kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros',
                        dropout=rnn_dropout_list[i], recurrent_dropout=rnn_r_dropout_list[i],
                        return_sequences=rnn_rt_seq[i], return_state=False
                    ),
                    merge_mode='concat', weights=None
                )
            )
        elif model == 'lstm':
            rnn_model.add(
                Bidirectional(
                    LSTM(
                        input_shape=rnn_in_shape_list[i],
                        units=rnn_unit_list[i],
                        activation=rnn_activation, recurrent_activation='hard_sigmoid',
                        use_bias=True, unit_forget_bias=True,
                        kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros',
                        dropout=rnn_dropout_list[i], recurrent_dropout=rnn_r_dropout_list[i],
                        return_sequences=rnn_rt_seq[i], return_state=False
                    ),
                    merge_mode='concat', weights=None
                )
            )

    # dnn layers
    for i in range(len(dnn_unit_list)):

        rnn_model.add(
            Dense(
                units=dnn_unit_list[i], activation=dnn_activation, use_bias=True,
                kernel_initializer='glorot_uniform', bias_initializer='zeros',
            )
        )
        if dnn_batch_norm:
            rnn_model.add(BatchNormalization())
        rnn_model.add( Dropout(rate=dnn_dropout_list[i]) )


    # output
    rnn_model.add(
        Dense(
            units=2, activation='softmax', use_bias=True,
            kernel_initializer='glorot_uniform', bias_initializer='zeros',
        )
    )

    return rnn_model


if __name__ == '__main__':


    args = ParseArgs()


    # w2v model
    ft_model = ft.load_fasttext_format(args.w2v)


    # training data
    x_train_total = LoadXTrain(ft_model, DEFAULT_W2V_DIM, args.x_train, str_max_len=DEFAULT_MAX_STR_LEN, oov=None)
    y_train_total = LoadYTrain(args.y_train)
    x_train, y_train, x_val, y_val = SplitValidate(x_train_total, y_train_total, args.validate, seed=DEFAULT_VAL_SEED)

    del x_train_total
    del y_train_total

    if args.x_semi and args.y_semi:
        x_train = np.concatenate((x_train, LoadXTrain(ft_model, DEFAULT_W2V_DIM, args.x_semi, 40, None)), axis=0)

    del ft_model

    if args.x_semi and args.y_semi:
        y_train = np.concatenate((y_train, LoadYTrain(args.y_semi)), axis=0)

    print ("Training data x shape: {}, Training ground truth y shape: {}".format(x_train.shape, y_train.shape))
    print ("Validation data x shape: {}, Validation ground truth y shape: {}".format(x_val.shape, y_val.shape))


    # rnn model
    rnn_model = DefineRNN(
                    model=args.model,
                    rnn_unit_list=DEFAULT_RNN_UNIT,
                    rnn_in_shape_list=DEFAULT_RNN_IN_SHAPE,
                    rnn_activation=args.rnn_activation,
                    rnn_dropout_list=DEFAULT_RNN_DROPOUT,
                    rnn_r_dropout_list=DEFAULT_RNN_R_DROPOUT,
                    rnn_rt_seq=DEFAULT_RNN_RT_SEQ,
                    dnn_unit_list=DEFAULT_DNN_UNIT,
                    dnn_activation=args.dnn_activation,
                    dnn_batch_norm=args.dnn_norm,
                    dnn_dropout_list=DEFAULT_DNN_DROPOUT
                )
    rnn_model.compile( optimizer=args.optimizer, loss='categorical_crossentropy', metrics=['accuracy'] )


    # callback functions
    check_point = ModelCheckpoint(
        filepath = args.out_model+"_epoch_{epoch:04d}_val_acc_{val_acc:.4f}_train_acc_{acc:.4f}.hdf5",
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


    # fit model
    rnn_model.fit(
        x=x_train, y=y_train,
        batch_size=args.batch_size,
        epochs=args.epoch,
        verbose=1,
        callbacks=[check_point, logger], # early_stop,
        validation_data=(x_val, y_val),
        shuffle=True,
        initial_epoch=0,
    )


    # save model
    print_summary(rnn_model)
    rnn_model.save(args.out_model+".hdf5")
