import argparse
import json
import os
from pathlib import Path
from graph import loss_acc_graph
import numpy as np

from model import build_model, save_weights, load_weights

DATA_DIR = './kdata'
LOG_DIR = './logs'
MODEL_DIR = './model'

BATCH_SIZE = 16
SEQ_LENGTH = 64


class TrainLogger(object):
    def __init__(self, file, resume=0):
        self.file = os.path.join(LOG_DIR, file)
        self.epochs = resume

        if not resume:
            with open(self.file, 'w') as f:
                f.write('epoch,loss,acc\n')

    def add_entry(self, loss, acc):
        self.epochs += 1
        s = '{},{},{}\n'.format(self.epochs, loss, acc)

        with open(self.file, 'a') as f:
            f.write(s)


def read_batches(t, vocab_size):
    length = t.shape[0]
    batch_chars = length // BATCH_SIZE

    for start in range(0, batch_chars - SEQ_LENGTH, SEQ_LENGTH):

        X = np.zeros((BATCH_SIZE, SEQ_LENGTH))
        Y = np.zeros((BATCH_SIZE, SEQ_LENGTH, vocab_size))

        for batch_idx in range(0, BATCH_SIZE):
            for i in range(0, SEQ_LENGTH):
                X[batch_idx, i] = t[batch_chars * batch_idx + start + i]
                Y[batch_idx, i, t[batch_chars * batch_idx + start + i + 1]] = 1

        yield X, Y


def train(train_text, epochs=100, save_freq=10, resume=False):
    total_loss, total_acc = [], []

    if resume:
        print("최근 학습에서 이어서 시작하기를 시도")

        model_dir = Path(MODEL_DIR)
        c2ifile = model_dir.joinpath('char_to_idx.json')

        with c2ifile.open('r') as f:
            char_to_idx = json.load(f)

        checkpoints = list(model_dir.glob('weights.*.h5'))

        if not checkpoints:
            raise ValueError("체크 포인트 확인 안됨")

        resume_epoch = max(int(p.name.split('.')[1]) for p in checkpoints)
        print("이어하기 시작 : ", resume_epoch)

    else:
        resume_epoch = 0
        char_to_idx = {ch: i for (i, ch) in enumerate(sorted(list(set(train_text))))}

        with open(os.path.join(MODEL_DIR, 'char_to_idx.json'), 'w') as f:
            json.dump(char_to_idx, f)

    vocab_size = len(char_to_idx)

    model = build_model(BATCH_SIZE, SEQ_LENGTH, vocab_size)
    model.summary()
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    if resume:
        total_loss, total_accs = load_weights(resume_epoch, model)

    T = np.asarray([char_to_idx[c] for c in train_text], dtype=np.int32)
    log = TrainLogger('training_log.csv', resume_epoch)

    for epoch in range(resume_epoch, epochs):
        print('\nEpoch {}/{}'.format(epoch + 1, epochs))

        losses, accs = [], []

        for i, (X, Y) in enumerate(read_batches(T, vocab_size)):
            loss, acc = model.train_on_batch(X, Y)

            total_loss.append(loss)
            total_acc.append(acc)

            print('Last {:.4f}% - step : {} | epoch : {} | Batch {} | loss = {:.4f}, acc = {:.5f}'.format(100 - (epoch * 100 / epochs), epoch * 143 + i + 1, epoch + 1, i + 1, loss, acc))

            losses.append(loss)
            accs.append(acc)

        log.add_entry(np.average(losses), np.average(accs))

        if (epoch + 1) % save_freq == 0:
            save_weights(epoch + 1, model, total_loss, total_acc)

            print('체크포인트 세이브', 'weights.{}.h5'.format(epoch + 1))

    loss_acc_graph(epochs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='txt를 학습함.')

    parser.add_argument('--input', default='sample.txt', help='학습시킬 파일 이름')
    parser.add_argument('--epochs', type=int, default=100, help='학습 시킬 epoch의 수')
    parser.add_argument('--freq', type=int, default=10, help='체크포인트 저장 빈도')
    parser.add_argument('--resume', action='store_true', help='학습을 이어서 진행하기')

    args = parser.parse_args()

    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    with open(os.path.join(DATA_DIR, args.input), 'r', encoding='utf8') as data_file:
        text = data_file.read()

    train(text, args.epochs, args.freq, args.resume)
