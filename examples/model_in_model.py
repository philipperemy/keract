import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import plot_model

import keract

# gradients requires no eager execution.
import utils

tf.compat.v1.disable_eager_execution()


def create_network_with_one_subnet():
    # FROM https://stackoverflow.com/questions/54648296/how-to-flatten-a-nested-model-keras-functional-api/54648506
    # define subnetwork
    subnet = keras.models.Sequential(name='subnet')
    subnet.add(keras.layers.Conv2D(6, (3, 3), padding='same'))
    subnet.add(keras.layers.MaxPool2D())
    subnet.add(keras.layers.Conv2D(12, (3, 3), padding='same'))
    subnet.add(keras.layers.MaxPool2D())
    # subnet.summary()

    # define complete network
    input_shape = (32, 32, 1)
    net_in = keras.layers.Input(shape=input_shape)
    net_out = subnet(net_in)
    net_out = keras.layers.Flatten()(net_out)
    net_out = keras.layers.Dense(1)(net_out)
    net_complete = keras.Model(inputs=net_in, outputs=net_out)
    net_complete.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(lr=0.001), metrics=['acc'])
    net_complete.summary()
    plot_model(net_complete)
    return net_complete


def main():
    np.random.seed(123)
    inputs = np.random.uniform(size=(8, 32, 32, 1))

    # Just for visual purposes.
    np.set_printoptions(precision=2)

    print('NESTED MODELS')
    nested_model = create_network_with_one_subnet()

    # will get the activations of every layer, EXCLUDING subnet layers.
    utils.print_names_and_values(keract.get_activations(nested_model, inputs))

    # will get the activations of every layer, including subnet.
    utils.print_names_and_values(keract.get_activations(nested_model, inputs, nested=True))


if __name__ == '__main__':
    main()
