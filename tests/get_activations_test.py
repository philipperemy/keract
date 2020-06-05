import unittest

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, concatenate

from keract import get_activations, get_gradients_of_activations, get_gradients_of_trainable_weights

tf.compat.v1.disable_eager_execution()


def dummy_model_and_inputs(**kwargs):
    i1 = Input(shape=(10,), name='i1')
    a = Dense(1, name='fc1')(i1)
    model = Model(inputs=[i1], outputs=[a], **kwargs)
    x = np.random.uniform(size=(32, 10))
    return model, x


class GetActivationsTest(unittest.TestCase):

    def setUp(self) -> None:
        K.clear_session()

    def tearDown(self) -> None:
        K.clear_session()

    def test_shape_1(self):
        # model definition
        i1 = Input(shape=(10,), name='i1')
        i2 = Input(shape=(10,), name='i2')

        a = Dense(1, name='fc1')(i1)
        b = Dense(1, name='fc2')(i2)

        c = concatenate([a, b], name='concat')
        d = Dense(1, name='out')(c)
        model = Model(inputs=[i1, i2], outputs=[d])

        # inputs to the model
        x = [np.random.uniform(size=(32, 10)),
             np.random.uniform(size=(32, 10))]

        # call to fetch the activations of the model.
        activations = get_activations(model, x, auto_compile=True)

        # OrderedDict so its ok to .values()
        self.assertListEqual([a.shape for a in activations.values()],
                             [(32, 10), (32, 10), (32, 1), (32, 1), (32, 2), (32, 1)])

    def test_auto_compile(self):
        model, x = dummy_model_and_inputs()

        def fun_to_test(auto_compile):
            get_activations(model, x, auto_compile=auto_compile)

        # should not work if auto_compile=False.
        self.assertRaises(Exception, fun_to_test, False)

        # should work if auto_compile=True.
        fun_to_test(True)

    def test_nodes_to_evaluate(self):
        model, x = dummy_model_and_inputs()

        w = model.layers[-1].kernel
        b = model.layers[-1].bias
        # not really activations here, just weight values. It's to show how to use it.
        weights_values = get_activations(model, x, nodes_to_evaluate=[w, b])

        wv = weights_values['fc1/kernel:0']
        bv = weights_values['fc1/bias:0']

        acts = get_activations(model, x)

        np.testing.assert_almost_equal(np.dot(x, wv) + bv, acts['fc1'], decimal=6)

    def test_layer_name(self):
        model, x = dummy_model_and_inputs()

        acts = get_activations(model, x, layer_name='fc1')
        self.assertListEqual(list(acts.keys()), ['fc1'])

        acts = get_activations(model, x, layer_name='i1')
        self.assertListEqual(list(acts.keys()), ['i1'])

        self.assertRaises(KeyError, lambda: get_activations(model, x, layer_name='unknown'))

    def test_output_format(self):
        model, x = dummy_model_and_inputs()

        simple = get_activations(model, x, output_format='simple')
        full = get_activations(model, x, output_format='full')
        numbered = get_activations(model, x, output_format='numbered')

        for s, f, n in zip(list(simple.values()), list(full.values()), list(numbered.values())):
            np.testing.assert_almost_equal(s, f)
            np.testing.assert_almost_equal(f, n)
            np.testing.assert_almost_equal(n, s)

        self.assertListEqual(list(simple.keys()), ['i1', 'fc1'])
        self.assertListEqual(list(full.keys()), ['i1:0', 'fc1/BiasAdd:0'])
        self.assertListEqual(list(numbered.keys()), [0, 1])

    def test_compile_vgg16_model(self):
        model, x = dummy_model_and_inputs(name="vgg16")
        get_activations(model, x, auto_compile=False)
        self.assertTrue(model._is_compiled)

    def test_nodes_and_layer_name(self):
        model, x = dummy_model_and_inputs()

        self.assertRaises(ValueError, lambda: get_activations(model,
                                                              x,
                                                              nodes_to_evaluate=[],
                                                              layer_name='unknown'))

    def test_nodes_empty(self):
        model, x = dummy_model_and_inputs()
        self.assertRaises(ValueError, lambda: get_activations(model, x, nodes_to_evaluate=[]))

    def test_gradients_of_activations(self):
        model, x = dummy_model_and_inputs()
        # important to leave the compile() call to the user. Gradients need this correct.
        model.compile(loss='mse', optimizer='adam')
        y = np.random.uniform(size=len(x))
        grad_acts = get_gradients_of_activations(model, x, y)
        acts = get_activations(model, x)

        # same support.
        self.assertListEqual(list(acts), list(grad_acts))
        self.assertListEqual(list(grad_acts['i1'].shape), list(acts['i1'].shape))
        self.assertListEqual(list(grad_acts['fc1'].shape), list(acts['fc1'].shape))

    def test_gradients_of_trainable_weights(self):
        model, x = dummy_model_and_inputs()
        model.compile(loss='mse', optimizer='adam')
        y = np.random.uniform(size=len(x))
        grad_trainable_weights = get_gradients_of_trainable_weights(model, x, y)

        self.assertListEqual(list(grad_trainable_weights), ['fc1/kernel:0', 'fc1/bias:0'])
        w = grad_trainable_weights['fc1/kernel:0']
        b = grad_trainable_weights['fc1/bias:0']
        self.assertListEqual(list(w.shape), [10, 1])  # Dense.w
        self.assertListEqual(list(b.shape), [1, ])  # Dense.b

    def test_inputs_order(self):
        i10 = Input(shape=(10,), name='i1')
        i40 = Input(shape=(40,), name='i4')
        i30 = Input(shape=(30,), name='i3')
        i20 = Input(shape=(20,), name='i2')

        a = Dense(1, name='fc1')(concatenate([i10, i40, i30, i20], name='concat'))
        model = Model(inputs=[i40, i30, i20, i10], outputs=[a])
        x = [
            np.random.uniform(size=(1, 40)),
            np.random.uniform(size=(1, 30)),
            np.random.uniform(size=(1, 20)),
            np.random.uniform(size=(1, 10))
        ]

        acts = get_activations(model, x)
        self.assertListEqual(list(acts['i1'].shape), [1, 10])
        self.assertListEqual(list(acts['i2'].shape), [1, 20])
        self.assertListEqual(list(acts['i3'].shape), [1, 30])
        self.assertListEqual(list(acts['i4'].shape), [1, 40])
