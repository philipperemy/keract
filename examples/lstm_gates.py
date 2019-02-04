import keras
from keras.layers import Dense
from keras.models import Sequential

import keract
import utils
from custom_lstm import LSTM
from data import get_mnist_data, num_classes

if __name__ == '__main__':
    x_train, y_train, _, _ = get_mnist_data()

    # (60000, 28, 28, 1) to ((60000, 28, 28)
    # LSTM has (batch, time_steps, input_dim)
    x_train = x_train.squeeze()

    lstm = LSTM(16, input_shape=(28, 28))

    model = Sequential()
    model.add(lstm)
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    import keras.backend as K

    K.eval(K.identity(lstm.cell.i, name='fd'))

    utils.print_names_and_shapes(keract.get_activations(model, x_train[:128]))
    utils.print_names_and_shapes(keract.get_gradients_of_trainable_weights(model, x_train[:128], y_train[:128]))
    utils.print_names_and_shapes(keract.get_gradients_of_activations(model, x_train[:128], y_train[:128]))
