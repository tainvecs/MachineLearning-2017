

import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import re


def ParseArgs():

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
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


    for file_name in os.listdir(args.log_dir):

        file_path = os.path.join(args.log_dir, file_name)
        out_name = "{}.loss_acc_curves.png".format(file_name.rsplit('.', 1)[0])
        out_path = os.path.join(args.output_dir, out_name)
        
        print (out_name)
        pd_log = pd.read_csv(file_path).iloc[:20,:]

        fig = plt.figure(figsize=(24, 10), dpi=250)

        str_m = re.match(r'.*_dnn_(selu|swish)_.*', out_name)
        atv_str = str_m.group(1)

        # Loss Curves
        ax = fig.add_subplot(1, 2, 1)
        ax.tick_params(labelsize=15)
        plt.plot(pd_log["epoch"], pd_log["loss"], "r", linewidth=0.8)
        plt.plot(pd_log["epoch"], pd_log["val_loss"], "b", linewidth=0.8)
        plt.legend(["Training Loss", "Validation Loss"], fontsize=15) 
        
        ax.set_xticks(np.arange(0, 21, 2))
        plt.ylim((0.0, 1))
        plt.xlabel("Epochs ", fontsize=15)
        plt.ylabel("Loss", fontsize=15)
        plt.title("Loss Curves ({})".format(atv_str), fontsize=15)

        # Accuracy Curves
        ax = fig.add_subplot(1, 2, 2)
        ax.tick_params(labelsize=15)
        plt.plot(pd_log["epoch"], pd_log["acc"], "r", linewidth=0.8)
        plt.plot(pd_log["epoch"], pd_log["val_acc"], "b", linewidth=0.8)
        plt.legend(["Training Accuracy", "Validation Accuracy"], fontsize=15)
        
        ax.set_xticks(np.arange(0, 21, 2))
        plt.ylim((0.6, 1.0))
        plt.xlabel("Epochs ", fontsize=15)
        plt.ylabel("Accuracy", fontsize=15)

        plt.title("Accuracy Curves ({})".format(atv_str), fontsize=15)

        plt.savefig(out_path, dpi=250, transparent=False)
