import argparse

import scipy.signal as signal
from matplotlib import pyplot as plt
import os
import numpy as np

BASE_DIR = ''


def arr_split_mean(arr, n):
    return [np.mean(arr[i * n:(i + 1) * n]) for i in range((len(arr) + n - 1) // n)]


def arr_in_arr_to_arr(arr):
    return [element[0] for element in arr]


def read_save(epoch, epoch_mean):
    file = open(os.path.join(BASE_DIR, 'logs/training_log.csv'), 'r', encoding='UTF-8')
    data = file.read().split('\n')[0:epoch]

    losses_accs = [list(map(float, loss_acc.split(','))) for one_line in data for loss_acc in one_line.split(' ')[:-1]]

    loss, acc = np.hsplit(np.asarray(losses_accs, dtype=np.float32), 2)

    loss = arr_in_arr_to_arr(loss)
    acc = arr_in_arr_to_arr(acc)

    if epoch_mean:
        loss = arr_split_mean(loss, int(len(loss) / epoch))
        acc = arr_split_mean(acc, int(len(acc) / epoch))

    return loss, acc, epoch_mean


def make_subplot(data, sub_locate, title, y_name, use_mean, use_medfilt):
    plt.subplot(sub_locate)

    x_data = list(range(1, len(data) + 1))
    y_data = signal.medfilt(data) if use_medfilt else data

    plt.plot(x_data, y_data)
    plt.title(title, color='skyblue', fontsize=30)

    plt.xlabel('epoch' if use_mean else 'step')
    plt.ylabel(y_name)


def loss_acc_graph(datas, use_medfilt):
    loss, acc, use_mean = datas

    make_subplot(loss, 211, 'Model Loss', 'loss', use_mean, use_medfilt)

    make_subplot(acc, 212, 'Model Acc', 'acc', use_mean, use_medfilt)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='학습 그래프 생성.')
    parser.add_argument('epoch', type=int, help='그래프에 사용할 epoch')
    parser.add_argument('--input', default='sample.txt', help='그래프를 그릴 파일 이름')
    parser.add_argument('--medfilt', action='store_false', help='medfilt 사용')
    parser.add_argument('--mean', action='store_false', help='epoch별 평균 사용')

    args = parser.parse_args()
    BASE_DIR = './' + args.input
    loss_acc_graph(read_save(args.epoch, args.mean), args.medfilt)
