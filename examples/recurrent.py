import keras
from keras.layers import Dense
from keras.layers.recurrent import LSTM
from keras.models import Sequential

import keract
import utils
from data import MNIST

if __name__ == '__main__':
    x_train, y_train, _, _ = MNIST.get_mnist_data()

    # (60000, 28, 28, 1) to ((60000, 28, 28)
    # LSTM has (batch, time_steps, input_dim)
    x_train = x_train.squeeze()

    model = Sequential()
    model.add(LSTM(16, input_shape=(28, 28)))
    model.add(Dense(MNIST.num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    utils.print_names_and_shapes(keract.get_activations(model, x_train[:128]))
    utils.print_names_and_shapes(keract.get_gradients_of_trainable_weights(model, x_train[:128], y_train[:128]))
    utils.print_names_and_shapes(keract.get_gradients_of_activations(model, x_train[:128], y_train[:128]))
