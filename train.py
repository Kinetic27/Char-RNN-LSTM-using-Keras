import argparse
import json
import os
import numpy as np
from pathlib import Path
from model import build_model, save_weights, load_weights
import time
import datetime

BASE_DIR = ''
DATA_DIR = './kdata'
LOG_DIR = '/logs'
MODEL_DIR = '/model'

BATCH_SIZE = 16
SEQ_LENGTH = 64


class TrainLogger(object):
    def __init__(self, file, resume=0):
        self.file = os.path.join(BASE_DIR + LOG_DIR, file)
        self.epochs = resume

        if not resume:
            open(self.file, 'w')

    def add_entry(self, loss, acc):
        with open(self.file, 'a') as f:
            f.write('{},{} '.format(loss, acc))

    def next_epoch(self, ):
        self.epochs += 1

        with open(self.file, 'a') as f:
            f.write('\n')


def safe_mkdir(file_dir):
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)


def read_batches(t, vocab_size):
    batch_chars = t.shape[0] // BATCH_SIZE

    for start in range(0, batch_chars - SEQ_LENGTH, SEQ_LENGTH):

        X = np.zeros((BATCH_SIZE, SEQ_LENGTH))
        Y = np.zeros((BATCH_SIZE, SEQ_LENGTH, vocab_size))

        for batch_idx in range(0, BATCH_SIZE):
            for i in range(0, SEQ_LENGTH):
                X[batch_idx, i] = t[batch_chars * batch_idx + start + i]
                Y[batch_idx, i, t[batch_chars * batch_idx + start + i + 1]] = 1

        yield X, Y


def train(train_text, epochs=100, save_freq=10, resume=False):
    resume_epoch = 0

    if resume:
        model_dir = Path(BASE_DIR + MODEL_DIR)
        c2ifile = model_dir.joinpath('char_to_idx.json')

        char_to_idx = json.load(c2ifile.open('r'))
        checkpoints = list(model_dir.glob('weights.*.h5'))

        if not checkpoints:
            raise ValueError("체크 포인트 확인 안됨")

        resume_epoch = max(int(p.name.split('.')[1]) for p in checkpoints)

    else:
        char_to_idx = {ch: i for (i, ch) in enumerate(sorted(list(set(train_text))))}
        json.dump(char_to_idx, open(os.path.join(BASE_DIR + MODEL_DIR, 'char_to_idx.json'), 'w'))

    vocab_size = len(char_to_idx)

    model = build_model(BATCH_SIZE, SEQ_LENGTH, vocab_size)
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    if resume:
        load_weights(BASE_DIR, resume_epoch, model)

    T = np.asarray([char_to_idx[c] for c in train_text], dtype=np.int32)
    log = TrainLogger('training_log.csv', resume_epoch)

    losses, accs = [], []
    old_time = time.time()

    for epoch in range(resume_epoch, epochs):
        tmp_losses, tmp_accs = [], []

        for i, (X, Y) in enumerate(read_batches(T, vocab_size)):
            loss, acc = model.train_on_batch(X, Y)

            tmp_losses.append(loss)
            tmp_accs.append(acc)

            batch_size = len(train_text) // BATCH_SIZE // SEQ_LENGTH  # 한 epoch는 몇 step?

            step = epoch * batch_size + i + 1  # 현재 몇 step 실행중?

            if step - resume_epoch * batch_size == 1:
                print("start!")
                old_time = time.time()

            last_per = 100 - step * 100 / (epochs * batch_size)  # 전체 남은 퍼센트

            resume_per = (step - resume_epoch * batch_size) * 100 / (
                    (epochs - resume_epoch) * batch_size)  # 재시작부터 남은 퍼센트
            last_sec = round(
                (time.time() - old_time) * (100 - resume_per) / resume_per)  # 남은 시간 = 걸린 시간 * 걸린 퍼센트 / 남은 퍼센트
            last_time_str = str(datetime.timedelta(seconds=last_sec))  # 남은 시간

            print('Last {:.4f}% (남은 시간 : {}) - step {} | epoch : {}/{} | Batch {} | loss = {:.4f}, acc = {:.5f}'.format(
                last_per, last_time_str, step, epoch + 1, epochs, i + 1, loss, acc
            ))

        losses.append(tmp_losses)
        accs.append(tmp_accs)

        if (epoch + 1) % save_freq == 0:
            save_weights(BASE_DIR, epoch + 1, model)

            for i in range(save_freq):
                for j in range(len(losses[0])):
                    log.add_entry(losses[i][j], accs[i][j])

                log.next_epoch()

            losses, accs = [], []

            print('체크포인트 세이브', 'weights.{}.h5'.format(epoch + 1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='txt를 학습함.')

    parser.add_argument('--input', default='sample.txt', help='학습시킬 파일 이름')
    parser.add_argument('--epochs', type=int, default=100, help='학습 시킬 epoch의 수')
    parser.add_argument('--freq', type=int, default=10, help='체크포인트 저장 빈도')
    parser.add_argument('--resume', action='store_true', help='학습을 이어서 진행하기')

    args = parser.parse_args()

    BASE_DIR = args.input

    safe_mkdir(BASE_DIR + LOG_DIR)
    safe_mkdir(BASE_DIR + MODEL_DIR)

    with open(os.path.join(DATA_DIR, args.input), 'r', encoding='utf8') as data_file:
        text = data_file.read()

    train(text, args.epochs, args.freq, args.resume)
