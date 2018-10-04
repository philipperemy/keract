import numpy as np
from keras.layers import merge, Dense
from keras.models import Input, Model, Sequential

from keract import get_activations


def get_multi_inputs_model():
    a = Input(shape=(10,))
    b = Input(shape=(10,))
    c = merge([a, b], mode='mul')
    c = Dense(1, activation='sigmoid', name='only_this_layer')(c)
    m_multi = Model(inputs=[a, b], outputs=c)
    return m_multi


def get_single_inputs_model():
    m_single = Sequential()
    m_single.add(Dense(1, activation='sigmoid', input_shape=(10,)))
    return m_single


if __name__ == '__main__':
    m = get_multi_inputs_model()
    m.compile(optimizer='adam',
              loss='binary_crossentropy')

    inp_a = np.random.uniform(size=(100, 10))
    inp_b = np.random.uniform(size=(100, 10))
    inp_o = np.random.randint(low=0, high=2, size=(100, 1))
    m.fit([inp_a, inp_b], inp_o)

    print('\n'.join([str(v.shape) for v in get_activations(m, [inp_a[0:1], inp_b[0:1]])]))
    get_activations(m, [inp_a[0:1], inp_b[0:1]], layer_name='only_this_layer')

    m2 = get_single_inputs_model()
    m2.compile(optimizer='adam',
               loss='binary_crossentropy')
    m2.fit([inp_a], inp_o)

    get_activations(m2, [inp_a[0]])
