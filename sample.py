import argparse
import json
import os
import numpy as np
from keras.layers import LSTM, Dropout, Dense, Activation, Embedding
from keras.models import Sequential
from model import load_weights

DATA_DIR = './kdata'
MODEL_DIR = 'model'
BASE_DIR = ''


def build_sample_model(vocab_size):
    model = Sequential()
    model.add(Embedding(vocab_size, 512, batch_input_shape=(1, 1)))

    for i in range(3):
        model.add(LSTM(256, return_sequences=(i != 2), stateful=True))
        model.add(Dropout(0.2))

    model.add(Dense(vocab_size))
    model.add(Activation('softmax'))

    return model


def sample(epoch, header, num_chars):
    with open(os.path.join(BASE_DIR, MODEL_DIR, 'char_to_idx.json'), 'r') as f:
        char_to_idx = json.load(f)

    idx_to_char = {i: ch for (ch, i) in list(char_to_idx.items())}
    vocab_size = len(char_to_idx)

    model = build_sample_model(vocab_size)
    load_weights(BASE_DIR, epoch, model)
    model.save(os.path.join(BASE_DIR, MODEL_DIR, 'model.{}.h5'.format(epoch)))

    sampled = [char_to_idx[c] for c in header]

    for c in header[:-1]:
        batch = np.zeros((1, 1))
        batch[0, 0] = char_to_idx[c]
        model.predict_on_batch(batch)

    for i in range(num_chars):
        batch = np.zeros((1, 1))
        if sampled:
            batch[0, 0] = sampled[-1]
        else:
            batch[0, 0] = np.random.randint(vocab_size)
        result = model.predict_on_batch(batch).ravel()
        sampleWord = np.random.choice(list(range(vocab_size)), p=result)
        sampled.append(sampleWord)

    return ''.join(idx_to_char[c] for c in sampled)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='모델에서 샘플을 뽑아냄')
    parser.add_argument('epoch', type=int, help='샘플을 뽑을 epoch')
    parser.add_argument('--input', default='sample.txt', help='샘플링시킬 파일 이름')
    parser.add_argument('--seed', default='', help='시작 단어 지정')
    parser.add_argument('--len', type=int, default=512, help='글자수 지정 (기본값 512)')
    args = parser.parse_args()

    BASE_DIR = args.input

    print(sample(args.epoch, args.seed, args.len))
