

import argparse

import numpy as np
import pandas as pd

import re


DEFAULT_VAL = 500
DEFAULT_SEED = None
DEFAULT_EPOCH = 10000
DEFAULT_BATCH = 50
DEFAULT_ETA = 0.01
DEFAULT_LAMBDA = 0.001
DEFAULT_OPT = 'adam'
DEFAULT_BETA_M = 0.9
DEFAULT_BETA_V = 0.999
DEFAULT_EPSILON = 1e-8
DEFAULT_NORM = 'standard'
DEFAULT_DEBUG = True
DEFAULT_EARLY_STOP = True
DEFAULT_EARLY_STOP_N = 100


def ParseArgs():

    parser = argparse.ArgumentParser()

    parser.add_argument('--train_feature', help='path of training feature; If both --train_feature and --train_answer are specified, new model will be trained and output.')
    parser.add_argument('--train_answer', help='path of training ground truth; If both --train_feature and --train_answer are specified, new model will be trained and output.')
    parser.add_argument('--test_feature', help='path of test feature; If specified, prediction will be output to --out_predict.')
    parser.add_argument('--test_answer', help='path of test answer')
    parser.add_argument('--validate', help='float 0~1: the proportions of validation set split from training dataset; int > 1: number of validation data slice from training dataset; validation data should not be more than 30\% of training dataset')
    parser.add_argument('--in_model', help='path of the model to load')

    parser.add_argument('--random_seed', help='random seed for splitting training and validation data')
    parser.add_argument('--epoch', help='number of training epoch')
    parser.add_argument('--batch_size', help='sgd mini batch size')
    parser.add_argument('--eta', help='learning rate')
    parser.add_argument('--l2_lambda', help='l2 norm lambda value')
    parser.add_argument('--optimizer', help='option: \"adam\", \"ada\"')
    parser.add_argument('--beta_m', help='bata value of momentum; The value should be specified if the optimizer is \"adam\".')
    parser.add_argument('--beta_v', help='bata value of velocity; The value should be specified if the optimizer is \"adam\".')
    parser.add_argument('--epsilon', help='The small value for avoiding divide by zero error while calculating gradient.')
    parser.add_argument('--norm', help='feature normalization option: none, standard, min_max, mean')
    parser.add_argument('--early_stop', help='early stopping: true or false')

    parser.add_argument('--out_log', help='path to output log file')
    parser.add_argument('--out_model', help='path to output model')
    parser.add_argument('--out_predict', help='path to output test prediction')
    parser.add_argument('--debug', help='option: true or false')

    args = parser.parse_args()


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

    if args.epoch:
        args.epoch = int(args.epoch)
    else:
        args.epoch = DEFAULT_EPOCH

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

    if args.norm.strip().lower() in ['none', 'standard', 'min_max', 'mean']:
        args.norm = args.norm.strip().lower()
    else:
        args.norm = DEFAULT_NORM

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
    args.args_str = 'epoch{}_early{}_val{}_seed{}_batch{}_eta{}_labmda{}_opt-{}_epsilon{}_norm-{}'.format(args.epoch,
        args.early_stop, args.validate, args.random_seed, args.batch_size, args.eta, args.l2_lambda, opt_str, args.epsilon, args.norm)

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


def __save_prediction(prediction, out_path):

    with open(out_path, 'w') as out_file:
        out_file.write("id,label\n")
        for i, num in enumerate(prediction):
            out_file.write("{},{}\n".format(i+1, num))


def __output_log(log, out_path):
    with open(out_path, 'a') as out_file:
        out_file.write(log)


class LogisticRegression():


    def __init__(self, epoch=10000, batch_size=50, eta=0.01, l2_lambda=0.001,
                 optimizer='adam', beta_m=0.9, beta_v=0.999, epsilon=1e-8, early_stop=True):

        self.epoch = int(epoch)
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
        self.accuracy = None
        self.performance_str = None


    def feature_normalization(self, feature_np, method, col_index_list, debug=True):

        if method == 'standard':

            feature_std = feature_np[:, col_index_list].std(axis=0)
            feature_mean = feature_np[:, col_index_list].mean(axis=0)

            stat_str = "Method: {}, standard deviation: {}, mean: {}".format(method, feature_std, feature_mean)
            feature_np[:, col_index_list] = (feature_np[:, col_index_list] - feature_mean) / feature_std

        elif method == 'min_max':

            feature_max = feature_np[:, col_index_list].max(axis=0)
            feature_min = feature_np[:, col_index_list].min(axis=0)

            stat_str = "Method: {}, max: {}, min: {}".format(method, feature_max, feature_min)
            feature_np[:, col_index_list] = (feature_np[:, col_index_list] - feature_min) / (feature_max - feature_min)

        elif method == 'mean':

            feature_mean = feature_np[:, col_index_list].mean(axis=0)
            feature_max = feature_np[:, col_index_list].max(axis=0)
            feature_min = feature_np[:, col_index_list].min(axis=0)

            stat_str = "Method: {}, mean: {}, max: {}, min: {}".format(method, feature_mean, feature_max, feature_min)
            feature_np[:, col_index_list] = (feature_np[:, col_index_list] - feature_mean) / (feature_max - feature_min)
        else:
            stat_str = "Method: None"

        if debug:
            print ("Feature Normalization {}".format(stat_str))

        return feature_np


    def split_validate(self, train_x, train_y, validate_n, seed=None, debug=True):

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


    def __sgd_batch_cost(self, class_prob, y, w, data_n):

        # cross entropy, l2 regularization
        cross_entropy = -1 * np.sum(np.dot(y, np.log(class_prob)) + np.dot((1-y.T), np.log(1-class_prob))) / data_n
        l2_reg = self.l2_lambda*np.sum(w**2) / (2 * data_n)

        return cross_entropy + l2_reg


    def __sgd_batch_gradient(self, x, w, error, data_n, iter_i):

        gradient = (np.dot(x.T, error) + self.l2_lambda*w) / data_n

        if self.optimizer == 'adam':

            self.momentum = self.beta_m*self.momentum + (1-self.beta_m)*gradient
            self.velocity = self.beta_v*self.velocity + (1-self.beta_v)*(gradient**2)

            momentum_hat = self.momentum / (1-self.beta_m**iter_i)
            velocity_hat = self.velocity / (1-self.beta_v**iter_i)

            adam_gradient = self.eta * momentum_hat / (np.sqrt(velocity_hat) + self.epsilon)

            return adam_gradient

        elif self.optimizer == 'ada':

            self.gradient_square += gradient**2
            ada_gradient = self.eta * gradient / (np.sqrt(self.gradient_square) + self.epsilon)

            return ada_gradient


    def __sigmoid(self, z):
        z_p = 1 / (1.0 + np.exp(-z))
        return np.clip(z_p, 1e-8, 1 - (1e-8))


    def __accuracy(self, prediction, answer, data_n):
        return np.sum(np.equal(prediction, answer)) / data_n


    def fit(self, x_train, y_train, x_val, y_val):

        # add bias term
        x_train = np.concatenate((x_train, np.ones((x_train.shape[0],1))), axis=1)
        x_val = np.concatenate((x_val, np.ones((x_val.shape[0],1))), axis=1)

        # w init with zero vector, the bias term b is the last term of wb
        # wb is the training target, the trained wb is the model
        para_n = x_train.shape[1]
        wb = np.zeros(para_n)

        # other variables
        b_mask = np.zeros(para_n)

        if self.optimizer == 'ada':
            self.gradient_square = np.zeros(para_n)
        elif self.optimizer == 'adam':
            self.momentum = np.zeros(para_n)
            self.velocity = np.zeros(para_n)

        accuracy_val_best = -float("inf")
        best_performance_info = ''
        patience_n = DEFAULT_EARLY_STOP_N

        # training process
        for epoch_i in range(self.epoch):

            # split sgd mini batch
            x_train_batch_list, y_train_batch_list = self.__sgd_shuffle_batch(x_train, y_train)

            accuracy_train_sum = 0
            cost_train_sum = 0

            # iteration
            for i, (x_train_batch, y_train_batch) in enumerate(zip(x_train_batch_list, y_train_batch_list)):

                iter_i = (epoch_i*len(x_train_batch_list)) + (i+1)
                data_n_train = x_train_batch.shape[0]

                # mask of the bias term; bias term is appended to the rear of w
                b_mask[-1] = -1 * wb[-1]

                # probability of binary prediction class 1
                class_prob_train = self.__sigmoid(np.dot(x_train_batch, wb))

                # calculate cost: cross entropy
                cost_train_sum += self.__sgd_batch_cost(class_prob_train, y_train_batch, (wb+b_mask), data_n_train)

                # training accuracy
                accuracy_train_sum += self.__accuracy(np.around(class_prob_train), y_train_batch, data_n_train)

                # calculate gradient
                gradient = self.__sgd_batch_gradient(x_train_batch, (wb+b_mask), (class_prob_train-y_train_batch), data_n_train, iter_i)

                # update
                wb = wb - gradient


            # train cost and train accuracy
            cost = cost_train_sum / len(x_train_batch_list)
            acc_train = accuracy_train_sum / len(x_train_batch_list)

            # validation accuracy
            class_prob_val = self.__sigmoid(np.dot(x_val, wb))
            accuracy_val = self.__accuracy(np.around(class_prob_val), y_val, x_val.shape[0])

            # performance info
            performance_info = 'Epoch {}, Cost {:.4f}, Training Accuracy {:.4f}, Validation Accuracy {:.4f}'.format(
                epoch_i, cost, acc_train, accuracy_val)

            if (epoch_i %10 == 0):
                print (performance_info, end='\r')

            # early stop
            if (self.early_stop) and (epoch_i > int(0.5*self.epoch)):

                if (accuracy_val >= accuracy_val_best):
                    patience_n = DEFAULT_EARLY_STOP_N
                else:
                    patience_n -= 1
                if patience_n == 0: break

            # update best
            if (accuracy_val >= accuracy_val_best):
                self.model = np.copy(wb)
                accuracy_val_best = accuracy_val
                best_performance_info = performance_info


        # best w, b, model
        if self.model is None:
            self.model = np.copy(wb)
            accuracy_val_best = accuracy_val
            best_performance_info = performance_info

        self.w, self.b = self.model[:-1], self.model[-1]
        self.accuracy = accuracy_val_best
        self.performance_str = best_performance_info

        print ('Best of {} Epochs: '.format(epoch_i)+best_performance_info)


        return self.w, self.b, self.model


    def predict(self, x, w, b):
        return np.where(self.__sigmoid(np.dot(x, w) + b) <= 0.5, 0, 1)


    def save_model(self, model, out_path):
        np.save(out_path, model)


    def load_model(self, model_path):

        self.model = np.load(model_path)
        self.w = self.model[:-1]
        self.b = self.model[-1]

        return self.w, self.b, self.model


if __name__ == '__main__':


    args = ParseArgs()


    logistic_reg_model = LogisticRegression(
        epoch=args.epoch, batch_size=args.batch_size, eta=args.eta, l2_lambda=args.l2_lambda,
        optimizer=args.optimizer, beta_m=args.beta_m, beta_v=args.beta_v, epsilon=args.epsilon, early_stop = args.early_stop
    )


    if args.train_feature and args.train_answer:


        # read csv by pandas, pd dataframe to np array
        if args.debug:
            print ("\n########## Reading Training Feature ###########")

        feature_train = pd.read_csv(args.train_feature).values.astype("float")
        answer_train = pd.read_csv(args.train_answer).values.flatten().astype("float")

        if args.debug:
            print ("Training data x shape: {}, Training ground truth y shape: {}".format(feature_train.shape, answer_train.shape))

        # preprocessing
        if args.debug:
            print ("\n####### Training Feature Preprocessing ########")

        # feature normalization
        feature_train = logistic_reg_model.feature_normalization(feature_train, args.norm, [0, 1, 3, 4, 5], debug=args.debug)

        # split train and validation
        if args.debug:
            print ("\n############ Split Validation data ############")
        x_train, y_train, x_val, y_val = logistic_reg_model.split_validate(
            feature_train, answer_train, args.validate, seed=args.random_seed, debug=args.debug)

        # train
        if args.debug:
            print ("\n############### Start Training ################")
        w, b, model = logistic_reg_model.fit(x_train, y_train, x_val, y_val)

        # save model
        if args.debug:
            print ("\n################# Save Model ##################")
            print (args.out_model)
        logistic_reg_model.save_model(model, args.out_model)

        # output log
        #log_str = "Args String: {}\nBest Accuracy: {}\n\n".format(args.args_str, logistic_reg_model.performance_str)
        log_str = "{},{},{},{},{},{}\n".format(
            args.optimizer, args.norm, args.batch_size, args.l2_lambda, args.eta, logistic_reg_model.accuracy)
        if args.debug:
            print ("\n############# Ouptut Training Log #############")
            print (log_str)
        __output_log(log_str, args.out_log)


    if args.test_feature:

        # read data
        if args.debug:
            print ("\n############ Reading Test Feature #############")
        feature_test = pd.read_csv(args.test_feature).values.astype("float")
        if args.debug:
            print ("Test data x shape: {}".format(feature_test.shape))

        # preprocessing
        if args.debug:
            print ("\n######### Test Feature Preprocessing ##########")

        # feature normalization
        feature_test = logistic_reg_model.feature_normalization(feature_test, args.norm, [0, 1, 3, 4, 5], debug=args.debug)

        # load model
        if args.in_model:
            w, b, model = logistic_reg_model.load_model(args.in_model)
            if args.debug:
                print ("\n################# Load Model ##################")
                print ("W shape: {}, b shape: {}".format(logistic_reg_model.w.shape, logistic_reg_model.b.shape))

        # predict
        prediction = None
        if not (logistic_reg_model.model is None) :
            if args.debug:
                print ("\n################ Predict Test #################")
            prediction = logistic_reg_model.predict(feature_test, w, b)

        if args.test_answer and not (prediction is None):
            answer_test = pd.read_csv(args.test_answer).iloc[:,1].values
            test_acc = np.sum( np.equal(prediction, answer_test)/answer_test.shape[0] )
            print ("Test Accuracy: {}".format(test_acc))

        # output prediction
        if not (prediction is None):
            __save_prediction(prediction, args.out_predict)
            if args.debug:
                print ("\n############## Output Prediction ##############")
                print (args.out_predict)
