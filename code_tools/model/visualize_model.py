import os
import logging
import numpy as np
import sys
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Helvetica'
plt.rcParams['font.size'] = 12

def draw_lr_cruve(mdl_path_list, figsize=(10, 6), linewidth=1, save=None):
    # plot
    fig = plt.figure(figsize=figsize)
    ax1 = fig.add_axes([0.1, 0.1, 0.4, 0.8])
    ax2 = fig.add_axes([0.57, 0.1, 0.4, 0.8])

    for mdl_dir in mdl_path_list:
        idx = '_'.join(mdl_dir.split('_')[3:]) 

        acc = np.load(os.path.join(mdl_dir, 'train_acc.npy'))
        val_acc = np.load(os.path.join(mdl_dir, 'val_acc.npy'))
        loss = np.load(os.path.join(mdl_dir, 'train_loss.npy'))
        val_loss = np.load(os.path.join(mdl_dir, 'val_loss.npy'))
        ax1.plot(acc, label=f'acc_{idx}', marker='o', linewidth=linewidth, markersize=1)
        ax1.plot(val_acc, label=f'val_acc_{idx}', marker='o', linewidth=linewidth, markersize=1)
        ax2.plot(loss, label=f'loss_{idx}', marker='o', linewidth=linewidth, markersize=1)
        ax2.plot(val_loss, label=f'val_loss_{idx}', marker='o', linewidth=linewidth, markersize=1)

    ax1.legend(); ax2.legend()
    ax1.set_xlabel('epochs'); ax2.set_xlabel('epochs')
    ax1.set_ylabel('accuracy'); ax2.set_xlabel('loss')
    if save:
        plt.savefig(save, dpi=100)
    plt.show()
    return fig



