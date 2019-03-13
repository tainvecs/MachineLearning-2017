

import argparse

import matplotlib.pyplot as plt
import pandas as pd


def ParseArgs():

    parser = argparse.ArgumentParser()

    parser.add_argument('--lr_log')
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
    lr = pd.read_csv(args.lr_log)

    # eta
    save_path = args.out_dir+'/eta-opt_{}_l2_lambda_{}_batch_size_{}_norm_{}.png'.format('adam', 0.0001, 10, 'standard')
    eta = lr[(lr.opt == 'adam')&(lr.batch == 10)&(lr.l2_lambda == 0.0001)&(lr.norm == 'standard')]\
        .sort_values(by=['eta'], ascending=False).reset_index(drop=True)
    plot_figure(eta['eta'], eta['acc'], 'Learning Rate (eta)', 'Accuracy (Validation)', title=None,
                x_lim=None, y_lim=[0.83, 0.86], x_log_scale = True, y_log_scale = False,
                show=False, save_path=save_path)

    # lambda
    save_path = args.out_dir+'/l2_lambda-opt_{}_eta_{}_batch_size_{}_norm_{}.png'.format('adam', 0.01, 10, 'standard')
    l2_lambda = lr[(lr.opt == 'adam')&(lr.batch == 10)&(lr.eta == 0.01)&(lr.norm == 'standard')]\
        .sort_values(by=['l2_lambda'], ascending=False).reset_index(drop=True)
    plot_figure(l2_lambda['l2_lambda'], l2_lambda['acc'],
                'L2 Regularization Lambda', 'Accuracy (Validation)', title=None,
                x_lim=None, y_lim=[0.83, 0.86], x_log_scale = True, y_log_scale = False,
                show=False, save_path=save_path)
