

import argparse

import matplotlib.pyplot as plt
import pandas as pd


def ParseArgs():

    parser = argparse.ArgumentParser()

    parser.add_argument('--adam_log')
    parser.add_argument('--ada_log')
    parser.add_argument('--out_dir')

    args = parser.parse_args()

    return args



def plot_figure(x, y, x_label, y_label, title=None, x_lim=None, y_lim=None,
    x_log_scale = False, y_log_scale = False, show=False, save_path=None):

    fig_pixel_x, fig_pixel_y = 1600, 900
    fig_dpi = 200

    fig, ax = plt.subplots(figsize=(fig_pixel_x/fig_dpi, fig_pixel_y/fig_dpi), dpi=fig_dpi)

    ax.plot(x, y, 'bo-', linewidth=0.7)

    ax.set(xlabel=x_label, ylabel=y_label)
    if title:
        ax.set(title=title)

    if x_lim:
        ax.set_xlim(x_lim)
    if y_lim:
        ax.set_ylim(y_lim)

    if x_log_scale:
        ax.set_xscale("log", nonposx='clip')
    if y_log_scale:
        ax.set_yscale("log", nonposy='clip')

    if save_path:
        fig.savefig(save_path, transparent=False, dpi=fig_dpi)

    if show:
        plt.show()



if __name__ == '__main__':


    args = ParseArgs()


    # input
    adam = pd.read_csv(args.adam_log)
    ada = pd.read_csv(args.ada_log)


    # adam
    # eta == 10, l2_lambda == 0.01, batch_size: 10, 50, 100, 500, 1000

    adam_slice = adam[(adam.eta == 10)&(adam.l2_lambda == 0.01)].sort_values(by=['batch_size'])
    y = adam_slice.reset_index(drop=True, inplace=False).rmse_val
    x = adam_slice.reset_index(drop=True, inplace=False).batch_size
    save_path = args.out_dir+'/adam-batch_size-eta_{}_l2_lambda_{}.png'.format(10, 0.01)
    plot_figure(x, y, x_label='Batch Size', y_label='RMSE (Validation)', title="Adam Optimizer",
                x_lim=(-50, 1050), y_lim=(0, 40), save_path=save_path)

    # adam
    # eta == 0.01, l2_lambda == 0.01, batch_size: 10, 50, 100, 500, 1000

    adam_slice = adam[(adam.eta == 0.01)&(adam.l2_lambda == 0.01)].sort_values(by=['batch_size'])
    y = adam_slice.reset_index(drop=True, inplace=False).rmse_val
    x = adam_slice.reset_index(drop=True, inplace=False).batch_size
    save_path = args.out_dir+'/adam-batch_size-eta_{}_l2_lambda_{}.png'.format(0.01, 0.01)
    plot_figure(x, y, x_label='Batch Size', y_label='RMSE (Validation)', title="Adam Optimizer",
                x_lim=(-50, 1050), y_lim=(5, 6), save_path=save_path)


    # ada
    # l2_lambda == 0.01, batch_size == 50, eta: 1e-4, 1e-3, 1e-2, 1e-1, 1, 10

    ada_slice = ada[(ada.l2_lambda == 0.01)&(ada.batch_size == 50)].sort_values(by=['eta'])
    y = ada_slice.reset_index(drop=True, inplace=False).rmse_val
    x = ada_slice.reset_index(drop=True, inplace=False).eta
    save_path = args.out_dir+'/ada-eta-batch_size_{}_l2_lambda_{}.png'.format(50, 0.01)
    plot_figure(x, y, x_label='Learning Rate (eta)', y_label='RMSE (Validation)', title="Ada Gradient",
                y_lim=(5, 10), x_log_scale=True, save_path=save_path)


    # adam
    # eta == 0.01, batch_size == 50, l2_lambda: 1e-4, 1e-3, 1e-2, 1e-1, 1, 10

    adam_slice = adam[(adam.eta == 0.01)&(adam.batch_size == 50)&(adam.l2_lambda != 0)].sort_values(by=['l2_lambda'])
    y = adam_slice.reset_index(drop=True, inplace=False).rmse_val
    x = adam_slice.reset_index(drop=True, inplace=False).l2_lambda
    save_path = args.out_dir+'/adam-l2_lambda-eta_{}_batch_size_{}.png'.format(0.01, 50)
    plot_figure(x, y, x_label='L2 Lambda', y_label='RMSE (Validation)', title="Adam Optimizer",
                y_lim=(5, 6), x_log_scale=True, save_path=save_path)


    # ada
    # eta == 0.01, batch_size == 50, l2_lambda: 1e-4, 1e-3, 1e-2, 1e-1, 1, 10

    ada_slice = ada[(ada.eta == 0.01)&(ada.batch_size == 50)&(ada.l2_lambda != 0)].sort_values(by=['l2_lambda'])
    y = ada_slice.reset_index(drop=True, inplace=False).rmse_val
    x = ada_slice.reset_index(drop=True, inplace=False).l2_lambda
    save_path = args.out_dir+'/ada-l2_lambda-eta_{}_batch_size_{}.png'.format(0.01, 50)
    plot_figure(x, y, x_label='L2 Lambda', y_label='RMSE (Validation)', title="Ada Gradient",
                y_lim=(5 , 6), x_log_scale=True, save_path=save_path)
