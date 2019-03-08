

import argparse

import numpy as np
import pandas as pd

import re


DEFAULT_SEED = None
DEFAULT_NORM = 'standard'
DEFAULT_DEBUG = True


def ParseArgs():

    parser = argparse.ArgumentParser()

    parser.add_argument('--train_x', help='path of training feature')
    parser.add_argument('--train_y', help='path of training ground truth')
    parser.add_argument('--validate', help='float 0~1: the proportions of validation set split from training dataset; int > 1: number of validation data slice from training dataset; validation data should not be more than 30\% of training dataset')
    parser.add_argument('--random_seed', help='random seed for splitting training and validation data')
    parser.add_argument('--norm', help='feature normalization option: none, standard, min_max, mean')
    parser.add_argument('--test_x', help='path of test feature')
    parser.add_argument('--test_y', help='path of test answer')
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
        args.validate = None

    if args.random_seed:
        args.random_seed = int(args.random_seed)
    else:
        args.random_seed = DEFAULT_SEED

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

    if args.debug:

        print ("################## Arguments ##################")

        args_dict = vars(args)
        for key in args_dict:
            print ( "\t{}: {}".format(key, args_dict[key]) )

        print ("###############################################")

    return args


class ProbabilisticGenerativeModel():


    def __init__(self):
        self.w = None
        self.b = None


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


    def fit(self, x, y):

        # class 0, class 1

        # index
        self.c0_index = np.where(y==0)[0]
        self.c1_index = np.where(y==1)[0]

        # x
        self.c0_x = x[self.c0_index, :]
        self.c1_x = x[self.c1_index, :]

        # mean
        self.c0_mean = np.mean(self.c0_x, axis=0)
        self.c1_mean = np.mean(self.c1_x, axis=0)

        # prior probability
        self.c0_pri_prob = len(self.c0_index) / (len(self.c0_index) + len(self.c1_index))
        self.c1_pri_prob = 1 - self.c0_pri_prob

        # covariance
        self.covariance = (np.cov(self.c0_x, rowvar=False)*self.c0_pri_prob) + (np.cov(self.c1_x, rowvar=False)*self.c1_pri_prob)
        self.covariance_inv = np.linalg.pinv(self.covariance)

        # w, b
        self.w = np.dot((self.c0_mean - self.c1_mean).T, self.covariance_inv)
        self.b = - 0.5 * np.dot( np.dot(self.c0_mean.T, self.covariance_inv), self.c0_mean ) \
                 + 0.5 * np.dot( np.dot(self.c1_mean.T, self.covariance_inv), self.c1_mean ) \
                 + np.log((len(self.c0_index)/len(self.c1_index)))

        return self.w, self.b

    def __sigmoid(self, z):
        z_p = 1 / (1.0 + np.exp(-z))
        return np.clip(z_p, 1e-8, 1 - (1e-8))

    def predict(self, x, w, b):
        return np.where(self.__sigmoid(np.dot(x, w) + b) > 0.5, 0, 1)


def __save_prediction(prediction, out_path):

    with open(out_path, 'w') as out_file:
        out_file.write("id,label\n")
        for i, num in enumerate(prediction):
            out_file.write("{},{}\n".format(i+1, num))


if __name__ == '__main__':


    args = ParseArgs()
    prob_gen_model = ProbabilisticGenerativeModel()


    # read csv by pandas, pd dataframe to np array
    if args.debug:
        print ("\n########## Reading Training Feature ###########")

    feature_train = pd.read_csv(args.train_x).values.astype("float")
    answer_train = pd.read_csv(args.train_y).values.flatten().astype("float")

    if args.debug:
        print ("Training data x shape: {}, Training ground truth y shape: {}".format(feature_train.shape, answer_train.shape))

    # preprocessing
    if args.debug:
        print ("\n####### Training Feature Preprocessing ########")

    # feature normalization
    feature_train = prob_gen_model.feature_normalization(feature_train, args.norm, [0, 1, 3, 4, 5], debug=args.debug)

    # split train and validation
    if not (args.validate is None):
        if args.debug:
            print ("\n############ Split Validation data ############")
        x_train, y_train, x_val, y_val = prob_gen_model.split_validate(
            feature_train, answer_train, args.validate, seed=args.random_seed, debug=args.debug)
    else:
        x_train, y_train, x_val, y_val = feature_train, answer_train, None, None

    # train
    w, b = prob_gen_model.fit(x_train, y_train)

    # validation accuracy
    if not (args.validate is None):
        prediction_val = prob_gen_model.predict(x_val, w, b)
        val_acc = np.sum( np.equal(prediction_val, y_val)/y_val.shape[0] )
        print ("Validation Accuracy: {}".format(val_acc))

    # read data
    if args.debug:
        print ("\n############ Reading Test Feature #############")
    feature_test = pd.read_csv(args.test_x).values.astype("float")
    if args.debug:
        print ("Test data x shape: {}".format(feature_test.shape))

    # preprocessing
    if args.debug:
        print ("\n######### Test Feature Preprocessing ##########")

    # feature normalization
    feature_test = prob_gen_model.feature_normalization(feature_test, args.norm, [0, 1, 3, 4, 5], debug=args.debug)

    # predict
    if args.debug:
        print ("\n################ Predict Test #################")
    prediction_test = prob_gen_model.predict(feature_test, w, b)

    if args.test_y:
        answer_test = pd.read_csv(args.test_y).iloc[:,1].values
        test_acc = np.sum( np.equal(prediction_test, answer_test)/answer_test.shape[0] )
        print ("Test Accuracy: {}".format(test_acc))

    if args.out_predict:
        __save_prediction(prediction_test, args.out_predict)
