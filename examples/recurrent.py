import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import Sequential

import keract
import utils
from data import MNIST

if __name__ == '__main__':
    # gradients requires no eager execution.
    tf.compat.v1.disable_eager_execution()
    # Check for GPUs and set them to dynamically grow memory as needed
    # Avoids OOM from tensorflow greedily allocating GPU memory
    utils.gpu_dynamic_mem_growth()

    x_train, y_train, _, _ = MNIST.get_mnist_data()

    # (60000, 28, 28, 1) to ((60000, 28, 28)
    # LSTM has (batch, time_steps, input_dim)
    x_train = x_train.squeeze()

    model = Sequential()
    model.add(LSTM(16, input_shape=(28, 28)))
    model.add(Dense(MNIST.num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    utils.print_names_and_shapes(keract.get_activations(model, x_train[:128]))
    utils.print_names_and_shapes(keract.get_gradients_of_trainable_weights(model, x_train[:128], y_train[:128]))
    utils.print_names_and_shapes(keract.get_gradients_of_activations(model, x_train[:128], y_train[:128]))
