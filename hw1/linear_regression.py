

import argparse

import numpy as np
import pandas as pd

import math
import re


DEFAULT_TRAIN = './data/train.csv'
DEFAULT_TEST = './data/test.csv'
DEFAULT_VAL = 500
DEFAULT_SEED = None
DEFAULT_ITER_N = 10000
DEFAULT_BATCH = 50
DEFAULT_ETA = 0.01
DEFAULT_LAMBDA = 0.001
DEFAULT_OPT = 'adam'
DEFAULT_BETA_M = 0.9
DEFAULT_BETA_V = 0.999
DEFAULT_EPSILON = 1e-8
DEFAULT_DEBUG = True
DEFAULT_EARLY_STOP = True
DEFAULT_EARLY_STOP_N = 100


def ParseArgs():

    parser = argparse.ArgumentParser()

    parser.add_argument('--train', help='path of training data; If specified, new model will be trained and output.')
    parser.add_argument('--test', help='path of test data; IIf specified, prediction will be output to --out_predict.')
    parser.add_argument('--validate', help='float 0~1: the proportions of validation set split from training dataset; int > 1: number of validation data slice from training dataset; validation data should not be more than 30\% of training dataset')
    parser.add_argument('--in_model', help='path of the model to load')

    parser.add_argument('--random_seed', help='random seed for splitting training and validation data')
    parser.add_argument('--iter_n', help='number of training iteration')
    parser.add_argument('--batch_size', help='sgd batch size')
    parser.add_argument('--eta', help='learning rate')
    parser.add_argument('--l2_lambda', help='l2 norm lambda value')
    parser.add_argument('--optimizer', help='option: \"adam\", \"ada\"')
    parser.add_argument('--beta_m', help='bata value of momentum; The value should be specified if the optimizer is \"adam\".')
    parser.add_argument('--beta_v', help='bata value of velocity; The value should be specified if the optimizer is \"adam\".')
    parser.add_argument('--epsilon', help='The small value for avoiding divide by zero error while calculating gradient.')
    parser.add_argument('--early_stop', help='early stopping: true or false')

    parser.add_argument('--out_log', help='path to output log file')
    parser.add_argument('--out_model', help='path to output model')
    parser.add_argument('--out_predict', help='path of test prediction')
    parser.add_argument('--debug', help='option: true or false')

    args = parser.parse_args()


    if not args.train:
        args.train = DEFAULT_TRAIN

    if not args.test:
        args.test = DEFAULT_TEST

    if args.validate:
        if re.match(r'[0-9]*$', args.validate):
            args.validate = int(args.validate)
        elif re.match(r'[0-9]*\.[0-9]*', args.validate):
            args.validate = float(args.validate)
        else:
            raise("Invalid Parameter: {}".format(args.validate))
    else:
        args.validate = DEFAULT_VAL

    if args.random_seed:
        args.random_seed = int(args.random_seed)
    else:
        args.random_seed = DEFAULT_SEED

    if args.iter_n:
        args.iter_n = int(args.iter_n)
    else:
        args.iter_n = DEFAULT_ITER_N

    if args.batch_size:
        args.batch_size = int(args.batch_size)
    else:
        args.batch_size = DEFAULT_BATCH

    if args.eta:
        args.eta = float(args.eta)
    else:
        args.eta = DEFAULT_ETA

    if args.l2_lambda:
        args.l2_lambda = float(args.l2_lambda)
    else:
        args.l2_lambda = DEFAULT_LAMBDA

    if args.optimizer:
        if args.optimizer.strip().lower() in ['adam', 'ada']:
            args.optimizer = args.optimizer.strip().lower()
        else:
            raise("Invalid optimizer: {}".format(args.optimizer))
    else:
        args.optimizer = DEFAULT_OPT

    if args.beta_m:
        args.beta_m = float(args.beta_m)
    else:
        args.beta_m = DEFAULT_BETA_M

    if args.beta_v:
        args.beta_v = float(args.beta_v)
    else:
        args.beta_v = DEFAULT_BETA_V

    if args.epsilon:
        args.epsilon = float(args.epsilon)
    else:
        args.epsilon = DEFAULT_EPSILON

    if args.early_stop:
        if args.early_stop.strip().lower() in ['true', 't', 'yes', 'y']:
            args.early_stop = True
        else:
            args.early_stop = False
    else:
        args.early_stop = DEFAULT_EARLY_STOP

    if args.debug:
        if args.debug.strip().lower() in ['true', 't', 'yes', 'y']:
            args.debug = True
        else:
            args.debug = False
    else:
        args.debug = DEFAULT_DEBUG

    if not args.out_log:
        args.out_log = './log/batch{}_eta{}_labmda{}_opt-{}.log'.format(
            args.batch_size, args.eta, args.l2_lambda, args.optimizer)

    if args.optimizer == 'adam':
        opt_str = "{}_batam{}_betav{}".format(args.optimizer, args.beta_m, args.beta_v)
    elif args.optimizer == 'ada':
        opt_str = args.optimizer
    args.args_str = 'iter{}_early{}_val{}_seed{}_batch{}_eta{}_labmda{}_opt-{}_epsilon{}'.format(args.iter_n,
        args.early_stop, args.validate, args.random_seed, args.batch_size, args.eta, args.l2_lambda, opt_str, args.epsilon)

    if not args.out_model:
        args.out_model = './model/{}.model'.format(args.args_str)

    if not args.out_predict:
        args.out_predict = './output/{}.predict'.format(args.args_str)


    if args.debug:

        print ("################## Arguments ##################")

        args_dict = vars(args)
        for key in args_dict:
            print ( "\t{}: {}".format(key, args_dict[key]) )

        print ("###############################################")

    return args


def __extract_train(in_path, debug=True):


    train = pd.read_csv(in_path)


    # replace no rain as 0
    train.replace('NR', 0, inplace=True)


    # slice the 18-row-data of each day and save it into 12 list according to the month
    # [[18*24 * 20] * 12]

    month_i = -1
    day_df_list = [[] for _ in range(12)]

    for i in range(0, train.shape[0], 18):

        if (i%360 == 0):
            month_i += 1

        day_df = train.iloc[i:i+18, 3:]
        day_df.reset_index(drop=True, inplace=True)
        day_df_list[month_i].append(day_df)


    # concat all the data in each month
    # [18 * 480] * 12

    month_df_list = []
    for m in range(12):
        month_df_list.append(pd.concat(day_df_list[m], axis=1, ignore_index=True, copy=True))
        month_df_list[m].index = train[:18].item


    # sliding window through the data of each month
    # extract training feature from 9 hours data and the 10th hour as ground thruth

    tmp_train_x, tmp_train_y = [], []
    for month_i in range(len(month_df_list)):
        for hour_i in range(0, month_df_list[month_i].shape[1]-9):
            tmp_train_x.append(month_df_list[month_i].iloc[:, hour_i:hour_i+9].values.reshape(1, 162)[0])
        tmp_train_y.append(month_df_list[month_i].loc['PM2.5', 9:])

    train_x = np.array(tmp_train_x, dtype=float)
    train_y = pd.concat(tmp_train_y, axis=0, ignore_index=True, copy=True).values.astype(int, copy=False)


    # delete data with incorrect ground truth -1
    # rm_idx: 52 index
    # train_x: (5600, 162)
    # train_y: (5600,)

    rm_idx = np.where(train_y < 0)
    train_x = np.delete(train_x, rm_idx, axis=0)
    train_y = np.delete(train_y, rm_idx, axis=0)


    # add bias term
    train_x = np.concatenate((train_x, np.ones((train_x.shape[0],1))), axis=1)

    if debug:
        print ("Training data x shape: {}, Training ground truth y shape: {}".format(train_x.shape, train_y.shape))

    return train_x, train_y


def __extract_test(in_path, debug=True):


    test = pd.read_csv(in_path, header=None)

    # replace no rain as 0
    test.replace('NR', 0, inplace=True)


    # slice the 18-row-data of each day
    # [18*24 * 20] * number_of_data

    tmp_test_x = []
    for i in range(0, test.shape[0], 18):

        day_df = test.iloc[i:i+18, 2:]
        tmp_test_x.append(day_df.values.reshape(1, 162)[0])

    test_x = np.array(tmp_test_x, dtype=float)

    if debug:
        print ("Test data x shape: {}".format(test_x.shape))

    return test_x


def __split_validate(train_x, train_y, validate_n, seed=None, debug=True):


    # val_n
    data_n = train_x.shape[0]
    if isinstance(validate_n, float):
        val_n = min(int(data_n*validate_n), int(data_n*0.3))
    elif isinstance(validate_n, int):
        val_n = min(int(validate_n), int(data_n*0.3))

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

    if debug:
        print ("Training data x shape: {}, Training ground truth y shape: {}".format(x_train.shape, y_train.shape))
        print ("Validation data x shape: {}, Validation ground truth y shape: {}".format(x_val.shape, y_val.shape))

    return x_train, y_train, x_val, y_val


def __save_prediction(prediction, out_path):

    with open(out_path, 'w') as out_file:
        for num in prediction:
            out_file.write("{}\n".format(num))


def __output_log(log, out_path):
    with open(out_path, 'a') as out_file:
        out_file.write(log)


class LinearRegression():


    def __init__(self, iter_n=10000, random_seed=None, batch_size=50, eta=0.01, l2_lambda=0.001,
                 optimizer='adam', beta_m=0.9, beta_v=0.999, epsilon=1e-8, early_stop=True):

        self.iter_n = int(iter_n)
        self.batch_size = int(batch_size)
        self.eta = float(eta)
        self.l2_lambda = float(l2_lambda)
        self.optimizer = optimizer.strip().lower()
        self.beta_m = float(beta_m)
        self.beta_v = float(beta_v)
        self.epsilon = float(epsilon)
        self.early_stop = early_stop

        self.w = None
        self.b = None
        self.model = None
        self.rmse = None
        self.rmse_str = None


    def __sgd_shuffle_batch(self, x_train, y_train):

        # shuffle index
        data_n = x_train.shape[0]
        s_idx = np.arange(data_n)
        np.random.shuffle(s_idx)

        # apply the shuffled index
        data_n = data_n - (data_n%self.batch_size)
        x_train = x_train[s_idx][:data_n]
        y_train = y_train[s_idx][:data_n]

        # split
        x_train_batch_list, y_train_batch_list = [], []
        for i in range(0, data_n, self.batch_size):
            x_train_batch_list.append(x_train[i:i+self.batch_size])
            y_train_batch_list.append(y_train[i:i+self.batch_size])

        return x_train_batch_list, y_train_batch_list


    def __sgd_batch_cost(self, loss_square, w, data_n):

        # l2 regularization
        return (loss_square + self.l2_lambda*np.sum(w**2)) / (2 * data_n)


    def __sgd_batch_gradient(self, x, w, loss, data_n):

        gradient = (np.dot(x.T, loss) + self.l2_lambda*w) / data_n

        if self.optimizer == 'adam':

            self.momentum = self.beta_m*self.momentum + (1-self.beta_m)*gradient
            self.velocity = self.beta_v*self.velocity + (1-self.beta_v)*(gradient**2)

            momentum_hat = self.momentum / (1-self.beta_m)
            velocity_hat = self.velocity / (1-self.beta_v)

            adam_gradient = self.eta * momentum_hat / (np.sqrt(velocity_hat) + self.epsilon)

            return adam_gradient

        elif self.optimizer == 'ada':

            self.gradient_square += gradient**2
            ada_gradient = self.eta * gradient / (np.sqrt(self.gradient_square) + self.epsilon)

            return ada_gradient


    def __RMSE(self, loss_square, data_n):
        return math.sqrt(loss_square / data_n)


    def fit(self, x_train, y_train, x_val, y_val):

        para_n = x_train.shape[1]

        # w init with zero vector, the bias term b is the last term of wb
        # wb is the training target, the trained wb is the model
        wb = np.zeros(para_n)

        # other variables
        b_mask = np.zeros(para_n)

        if self.optimizer == 'ada':
            self.gradient_square = np.zeros(para_n)
        elif self.optimizer == 'adam':
            self.momentum = np.zeros(para_n)
            self.velocity = np.zeros(para_n)

        RMSE_val_best = float("inf")
        RMSE_val_best_info = ''
        patience_n = DEFAULT_EARLY_STOP_N

        # training process
        for iter_i in range(self.iter_n):

            # split sgd mini batch
            x_train_batch_list, y_train_batch_list = self.__sgd_shuffle_batch(x_train, y_train)

            RMSE_train_sum = 0
            cost_sum = 0
            for x_train_batch, y_train_batch in zip(x_train_batch_list, y_train_batch_list):

                data_n_train = x_train_batch.shape[0]

                # mask of the bias term
                # bias term is append to the rear of w
                b_mask[-1] = -1 * wb[-1]

                # loss, cost, RMSE
                loss_train = np.dot(x_train_batch, wb) - y_train_batch
                loss_square_train = np.sum(loss_train**2)

                # calculate cost
                cost_sum += self.__sgd_batch_cost(loss_square_train, (wb+b_mask), data_n_train)

                # train RMSE
                RMSE_train_sum += self.__RMSE(loss_square_train, data_n_train)

                # calculate gradient
                gradient = self.__sgd_batch_gradient(x_train_batch, (wb+b_mask), loss_train, data_n_train)

                # update
                wb = wb - gradient

            # validation
            loss_val = np.dot(x_val, wb) - y_val
            loss_square_val = np.sum(loss_val**2)
            RMSE_val = self.__RMSE(loss_square_val, x_val.shape[0])

            # RMSE_info
            cost = cost_sum / len(x_train_batch_list)
            RMSE_train = RMSE_train_sum / len(x_train_batch_list)
            RMSE_val_info = 'Iteration {}, Cost {:.4f}, Train RMSE {:.4f}, Validation RMSE {:.4f}'.format(
                iter_i, cost, RMSE_train, RMSE_val)

            if (iter_i %10 == 0):
                print (RMSE_val_info, end='\r')

            # early stop
            if (self.early_stop) and (iter_i > int(0.5*self.iter_n)):

                if (RMSE_val < RMSE_val_best):
                    patience_n = DEFAULT_EARLY_STOP_N
                else:
                    patience_n -= 1
                if patience_n == 0: break

            # update best
            if (RMSE_val < RMSE_val_best):
                self.model = np.copy(wb)
                RMSE_val_best = RMSE_val
                RMSE_val_best_info = RMSE_val_info


        # best w, b, model
        if self.model is None:
            self.model = np.copy(wb)
            RMSE_val_best = RMSE_val
            RMSE_val_best_info = RMSE_val_info

        self.w, self.b = self.model[:-1], self.model[-1]
        self.rmse = RMSE_val_best
        self.rmse_str = RMSE_val_best_info

        print ('Best of {} iterations: '.format(iter_i)+RMSE_val_best_info)


        return self.w, self.b, self.model


    def predict(self, x, w, b):
        return np.dot(x, w) + b


    def save_model(self, model, out_path):
        np.save(out_path, model)


    def load_model(self, model_path):

        self.model = np.load(model_path)
        self.w = self.model[:-1]
        self.b = self.model[-1]

        return self.w, self.b, self.model


if __name__ == '__main__':


    args = ParseArgs()


    linear_reg_model = LinearRegression(
        iter_n=args.iter_n, random_seed=args.random_seed, batch_size=args.batch_size, eta=args.eta, l2_lambda=args.l2_lambda,
        optimizer=args.optimizer, beta_m=args.beta_m, beta_v=args.beta_v, epsilon=args.epsilon, early_stop = args.early_stop
    )


    if args.train:

        # extract
        if args.debug:
            print ("\n############ Extract Training data ############")
        train_x_all, train_y_all = __extract_train(args.train, debug=args.debug)

        # split train and validation
        if args.debug:
            print ("\n############ Split Validation data ############")
        x_train, y_train, x_val, y_val = __split_validate(train_x_all, train_y_all, args.validate, seed=args.random_seed, debug=args.debug)

        # train
        if args.debug:
            print ("\n############### Start Training ################")
        w, b, model = linear_reg_model.fit(x_train, y_train, x_val, y_val)

        # save model
        linear_reg_model.save_model(model, args.out_model)
        if args.debug:
            print ("\n################# Save Model ##################")
            print (args.out_model)

        # output log
        #log_str = "Args String: {}\nBest RMSE: {}\n\n".format(args.args_str, linear_reg_model.rmse_str)
        log_str = "{}, {}, {}, {}\n".format(args.batch_size, args.l2_lambda, args.eta, linear_reg_model.rmse)
        __output_log(log_str, args.out_log)
        if args.debug:
            print ("\n############# Ouptut Training Log #############")
            print (log_str)


    if args.test:

        # extract data
        if args.debug:
            print ("\n############## Extract Test data ##############")
        x_test = __extract_test(args.test, debug=args.debug)

        # load model
        if args.in_model:
            linear_reg_model.load_model(args.in_model)
            if args.debug:
                print ("\n################# Load Model ##################")
                print ("W shape: {}, b shape: {}".format(linear_reg_model.w.shape, linear_reg_model.b.shape))

        # predict
        prediction = None
        if not (linear_reg_model.model is None) :
            if args.debug:
                print ("\n################ Predict Test #################")
            prediction = linear_reg_model.predict(x_test, linear_reg_model.w, linear_reg_model.b)

        # output prediction
        if not (prediction is None):
            __save_prediction(prediction, args.out_predict)
            if args.debug:
                print ("\n############## Output Prediction ##############")
                print (args.out_predict)
