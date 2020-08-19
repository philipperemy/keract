import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Add, Dense, Input
from tensorflow.keras.models import Model

import keract
# gradients requires no eager execution.
import utils

tf.compat.v1.disable_eager_execution()


def get_multi_outputs_model():
    a = Input(shape=(10,))
    b = Input(shape=(10,))
    c = Add()([a, b])
    d = Dense(1, activation='sigmoid', name='o1')(c)
    e = Dense(1, activation='sigmoid', name='o2')(c)
    m_multi = Model(inputs=[a, b], outputs=[d, e])
    # plot_model(m_multi)
    return m_multi


def main():
    np.random.seed(123)
    inp_a = np.random.uniform(size=(5, 10))
    inp_b = np.random.uniform(size=(5, 10))
    out_d = np.random.uniform(size=(5, 1))
    out_e = np.random.uniform(size=(5, 1))

    # Just for visual purposes.
    np.set_printoptions(precision=2)

    # Activations of all the layers
    print('MULTI-INPUT OUTPUT MODEL')
    m1 = get_multi_outputs_model()
    m1.compile(optimizer='adam', loss='mse')
    m1.fit(x=[inp_a, inp_b], y=[out_d, out_e])

    utils.print_names_and_values(keract.get_activations(m1, [inp_a, inp_b]))


if __name__ == '__main__':
    main()
