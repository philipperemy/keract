import os
import unittest

import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dropout, Dense

from keract import get_activations, persist_to_json_file, load_activations_from_json_file


class PersistLoadTest(unittest.TestCase):

    def setUp(self) -> None:
        K.clear_session()

    def tearDown(self) -> None:
        K.clear_session()
        os.remove('activations.json')

    def test_load_persist(self):
        # define the model.
        model = Sequential()
        model.add(Dense(16, input_shape=(10,)))
        model.add(Dropout(0.5))
        model.add(Dense(10, activation='softmax'))
        model.compile(optimizer='adam', loss='categorical_crossentropy')

        # fetch activations.
        x = np.ones((2, 10))
        activations = get_activations(model, x)

        # persist the activations to the disk.
        output = 'activations.json'
        persist_to_json_file(activations, output)

        # read them from the disk.
        activations2 = load_activations_from_json_file(output)

        for a1, a2 in zip(list(activations.values()), list(activations2.values())):
            np.testing.assert_almost_equal(a1, a2)
