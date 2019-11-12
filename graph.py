import argparse

import scipy.signal as signal
from matplotlib import pyplot as plt
import os
import numpy as np


def read_save(folder, file_name, epoch, one_batch, epoch_mean=True):
    file = open(os.path.join(folder, '{}.{}.txt'.format(file_name, epoch)), 'r', encoding='UTF-8')
    string_data = file.read()
    file.close()

    data_array = list(map(float, string_data.split(', ')))

    if epoch_mean:
        data_array = [np.mean(data_array[i * one_batch:(i + 1) * one_batch - 1]) for i in range(epoch)]

    return data_array, epoch_mean


def make_subplot(data, sub_locate, title, y_name, use_medfilt=True):
    plt.subplot(sub_locate)

    x_data = list(range(1, len(data[0]) + 1))
    y_data = signal.medfilt(data[0]) if use_medfilt else data[0]

    plt.plot(x_data, y_data)
    plt.title(title, color='skyblue', fontsize=30)

    plt.xlabel('epoch' if data[1] else 'step')
    plt.ylabel(y_name)


def loss_acc_graph(epoch):
    make_subplot(
        read_save('./loss', 'losses', epoch, 143),
        211, 'Model Loss', 'loss'
    )

    make_subplot(
        read_save('./acc', 'accs', epoch, 143),
        212, 'Model Acc', 'acc'
    )

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='학습 그래프 생성.')
    parser.add_argument('epoch', type=int, help='그래프에 사용할 epoch')
    args = parser.parse_args()

    loss_acc_graph(args.epoch)
