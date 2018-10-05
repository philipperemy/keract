import keras
from keras.layers import Dense
from keras.layers.recurrent import LSTM
from keras.models import Sequential

import utils
from data import get_mnist_data, num_classes
from keract import get_activations

if __name__ == '__main__':
    x_train, _, _, _ = get_mnist_data()

    # (60000, 28, 28, 1) to ((60000, 28, 28)
    # LSTM has (batch, time_steps, input_dim)
    x_train = x_train.squeeze()

    model = Sequential()
    model.add(LSTM(16, input_shape=(28, 28)))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    utils.print_names_and_shapes(get_activations(model, x_train))
