import os

from keras.layers import LSTM, Dropout, TimeDistributed, Dense, Activation, Embedding
from keras.models import Sequential
from keras.utils import plot_model


MODEL_DIR = '/model'


def save_weights(base, epoch, s_model):
    if not os.path.exists(base + MODEL_DIR):
        os.makedirs(base + MODEL_DIR)

    s_model.save_weights(os.path.join(base + MODEL_DIR, 'weights.{}.h5'.format(epoch)))


def load_weights(base, epoch, l_model):
    l_model.load_weights(os.path.join(base + MODEL_DIR, 'weights.{}.h5'.format(epoch)))


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
