import os

from keras.layers import LSTM, Dropout, TimeDistributed, Dense, Activation, Embedding
from keras.models import Sequential
from keras.utils import plot_model


MODEL_DIR = './model'
ACC_DIR = './acc'
LOSS_DIR = './loss'


def save_weights(epoch, s_model, losses, accs):
    folders = [MODEL_DIR, ACC_DIR, LOSS_DIR]

    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)

    s_model.save_weights(os.path.join(MODEL_DIR, 'weights.{}.h5'.format(epoch)))

    l_file = open(os.path.join(LOSS_DIR, 'losses.{}.txt'.format(epoch)), 'w')
    l_file.write(', '.join(map(str, losses)))
    l_file.close()

    a_file = open(os.path.join(ACC_DIR, 'accs.{}.txt'.format(epoch)), 'w')
    a_file.write(', '.join(map(str, accs)))
    a_file.close()


def load_weights(epoch, l_model):
    l_model.load_weights(os.path.join(MODEL_DIR, 'weights.{}.h5'.format(epoch)))

    loss_txt = open(os.path.join(LOSS_DIR, 'losses.{}.txt'.format(epoch)), 'r', encoding='UTF-8')
    loss_array = loss_txt.read().split(', ')
    loss_txt.read()

    acc_txt = open(os.path.join(ACC_DIR, 'accs.{}.txt'.format(epoch)), 'r', encoding='UTF-8')
    acc_array = acc_txt.read().split(', ')
    acc_txt.read()

    return loss_array, acc_array


def build_model(batch_size, seq_len, vocab_size):
    b_model = Sequential([
        Embedding(vocab_size, 512, batch_input_shape=(batch_size, seq_len)),

        LSTM(256, return_sequences=True, stateful=True),
        Dropout(0.2),

        LSTM(256, return_sequences=True, stateful=True),
        Dropout(0.2),

        LSTM(256, return_sequences=True, stateful=True),
        Dropout(0.2),

        TimeDistributed(Dense(vocab_size)),
        Activation('softmax')
    ])

    return b_model


if __name__ == '__main__':
    model = build_model(16, 64, 50)
    model.summary()

    plot_model(model, to_file='model.png')

    print('사진 저장!')
