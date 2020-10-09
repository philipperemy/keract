import os
import unittest
from glob import glob

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense

from keract import get_activations, display_activations
from keract.keract import _convert_1d_to_2d

tf.compat.v1.disable_eager_execution()


def dummy_model_and_inputs():
    i1 = Input(shape=(10,), name='i1')
    a = Dense(1, name='fc1')(i1)
    model = Model(inputs=[i1], outputs=[a])
    x = np.random.uniform(size=(1, 10))
    return model, x


class DisplayActivationsTest(unittest.TestCase):

    def setUp(self) -> None:
        K.clear_session()

    def tearDown(self) -> None:
        K.clear_session()
        for image in glob('*.png'):
            os.remove(image)

    def test_display_1(self):
        model, x = dummy_model_and_inputs()
        acts = get_activations(model, x)
        display_activations(acts, save=True)

    def test_display_2(self):
        acts = {'1_channel': np.random.uniform(size=(1, 32, 32, 1))}
        display_activations(acts, save=True)

    def test_convert_1d_to_2d(self):
        self.assertEqual(_convert_1d_to_2d(64), (8, 8))
        self.assertEqual(_convert_1d_to_2d(32), (8, 4))
        self.assertEqual(_convert_1d_to_2d(32 * 33), (33, 32))
