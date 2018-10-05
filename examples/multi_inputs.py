import numpy as np
from keras.layers import Add, Dense
from keras.models import Input, Model

import utils
from keract import get_activations


def get_multi_inputs_model():
    a = Input(shape=(10,))
    b = Input(shape=(10,))
    c = Add()([a, b])
    c = Dense(1, activation='sigmoid', name='last_layer')(c)
    m_multi = Model(inputs=[a, b], outputs=c)
    return m_multi


def get_single_inputs_model():
    inputs = Input(shape=(10,))
    x = Dense(1, activation='sigmoid')(inputs)
    m_single = Model(inputs=[inputs], outputs=x)
    return m_single


if __name__ == '__main__':
    np.random.seed(123)
    inp_a = np.random.uniform(size=(5, 10))
    inp_b = np.random.uniform(size=(5, 10))

    # Just for visual purposes.
    np.set_printoptions(precision=2)

    # Activations of all the layers
    print('MULTI-INPUT MODEL')
    m1 = get_multi_inputs_model()
    utils.print_names_and_values(get_activations(m1, [inp_a, inp_b]))

    # Just get the last layer!
    print(get_activations(m1, [inp_a, inp_b], layer_name='last_layer'))
    print('')

    print('SINGLE-INPUT MODEL')
    m2 = get_single_inputs_model()
    utils.print_names_and_values(get_activations(m2, inp_a))
