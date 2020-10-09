import unittest

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.layers import Add, Dense, Input
from tensorflow.keras.layers import ReLU, Layer, concatenate

from keract import get_activations, get_gradients_of_activations, get_gradients_of_trainable_weights, keract



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
    # plot_model(net_complete)
    return net_complete


def dummy_model_and_inputs(**kwargs):
    i1 = Input(shape=(10,), name='i1')
    a = NestedModel(name='model')(i1)
    b = NestedLayer(name='block')(a)
    c = Dense(1, name='fc1')(b)
    model = Model(inputs=[i1], outputs=[c], **kwargs)
    x = np.random.uniform(size=(32, 10))
    return model, x


def get_multi_outputs_model():
    a = Input(shape=(10,), name='i1')
    b = Input(shape=(10,), name='i2')
    c = Add(name='add')([a, b])
    d = Dense(1, activation='sigmoid', name='o1')(c)
    e = Dense(2, activation='sigmoid', name='o2')(c)
    m_multi = Model(inputs=[a, b], outputs=[d, e])
    # plot_model(m_multi)
    return m_multi


class NestedModel(Model):
    def __init__(self, *args, **kwargs):
        super(NestedModel, self).__init__(*args, **kwargs)
        self.fc = Dense(10, name='fc1')
        self.relu = ReLU(name='relu')

    def call(self, x):
        return self.relu(self.fc(x))


class NestedLayer(Layer):
    def __init__(self, *args, **kwargs):
        super(NestedLayer, self).__init__(*args, **kwargs)
        self.fc = Dense(10, name='fc1')
        self.relu = ReLU(name='relu')

    def call(self, x):
        return self.relu(self.fc(x))


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

        z = model.layers[2].output
        w = model.layers[-1].kernel
        b = model.layers[-1].bias
        # not really activations here, just weight values. It's to show how to use it.
        weights_values = get_activations(model, x, nodes_to_evaluate=[z, w, b])

        print("Weights valuese:", weights_values)
        zv = weights_values['block/relu/Relu:0']
        wv = weights_values['fc1/kernel:0']
        bv = weights_values['fc1/bias:0']

        acts = get_activations(model, x)

        np.testing.assert_almost_equal(np.dot(zv, wv) + bv, acts['fc1'], decimal=6)

    def test_layer_name(self):
        model, x = dummy_model_and_inputs()

        acts = get_activations(model, x, layer_names='fc1')
        self.assertListEqual(list(acts.keys()), ['fc1'])

        acts = get_activations(model, x, layer_names='model')
        self.assertListEqual(list(acts.keys()), ['model'])

        acts = get_activations(model, x, layer_names='block')
        self.assertListEqual(list(acts.keys()), ['block'])

        acts = get_activations(model, x, layer_names=['block/fc1', 'block/relu'], nested=True)
        self.assertListEqual(list(acts.keys()), ['block/fc1', 'block/relu'])

        acts = get_activations(model, x, layer_names=['model/fc1', 'model/relu'], nested=True)
        self.assertListEqual(list(acts.keys()), ['model/fc1', 'model/relu'])

        acts = get_activations(model, x, layer_names='i1')
        self.assertListEqual(list(acts.keys()), ['i1'])

        self.assertRaises(KeyError, lambda: get_activations(model, x, layer_names='unknown'))

    def test_output_format(self):
        model, x = dummy_model_and_inputs()

        simple = get_activations(model, x, output_format='simple')
        simple_nested = get_activations(model, x, output_format='simple', nested=True)
        full = get_activations(model, x, output_format='full')
        full_nested = get_activations(model, x, output_format='full', nested=True)
        numbered = get_activations(model, x, output_format='numbered')

        for s, f, n in zip(list(simple.values()), list(full.values()), list(numbered.values())):
            np.testing.assert_almost_equal(s, f)
            np.testing.assert_almost_equal(f, n)
            np.testing.assert_almost_equal(n, s)

        self.assertListEqual(list(simple.keys()), ['i1', 'model', 'block', 'fc1'])
        self.assertListEqual(list(simple_nested.keys()), ['i1',
                                                          'model/fc1',
                                                          'model/relu',
                                                          'block/fc1',
                                                          'block/relu',
                                                          'fc1'])
        self.assertListEqual(list(full.keys()), ['i1:0', 'model/relu/Relu:0', 'block/relu/Relu:0', 'fc1/BiasAdd:0'])
        self.assertListEqual(list(full_nested.keys()), ['i1:0',
                                                        'model/fc1/BiasAdd:0',
                                                        'model/relu/Relu:0',
                                                        'block/fc1/BiasAdd:0',
                                                        'block/relu/Relu:0',
                                                        'fc1/BiasAdd:0'])
        self.assertListEqual(list(numbered.keys()), [0, 1, 2, 3])

    def test_compile_vgg16_model(self):
        model, x = dummy_model_and_inputs(name="vgg16")
        get_activations(model, x, auto_compile=False)
        self.assertTrue(model._is_compiled)

    def test_nodes_and_layer_name(self):
        model, x = dummy_model_and_inputs()

        self.assertRaises(ValueError, lambda: get_activations(model,
                                                              x,
                                                              nodes_to_evaluate=[],
                                                              layer_names='unknown'))

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

        grad_acts_nested = get_gradients_of_activations(model, x, y, nested=True)
        acts_nested = get_activations(model, x, nested=True)

        # same support.
        self.assertListEqual(list(acts), list(grad_acts))
        self.assertListEqual(list(grad_acts['i1'].shape), list(acts['i1'].shape))
        self.assertListEqual(list(grad_acts['model'].shape), list(acts['model'].shape))
        self.assertListEqual(list(grad_acts['block'].shape), list(acts['block'].shape))
        self.assertListEqual(list(grad_acts['fc1'].shape), list(acts['fc1'].shape))

        self.assertListEqual(list(acts_nested), list(grad_acts_nested))
        self.assertListEqual(list(grad_acts_nested['i1'].shape), list(acts_nested['i1'].shape))
        self.assertListEqual(list(grad_acts_nested['model/fc1'].shape), list(acts_nested['model/fc1'].shape))
        self.assertListEqual(list(grad_acts_nested['model/fc1'].shape), list(acts_nested['model/fc1'].shape))
        self.assertListEqual(list(grad_acts_nested['block/fc1'].shape), list(acts_nested['block/fc1'].shape))
        self.assertListEqual(list(grad_acts_nested['block/relu'].shape), list(acts_nested['block/relu'].shape))
        self.assertListEqual(list(grad_acts_nested['fc1'].shape), list(acts_nested['fc1'].shape))

    def test_gradients_of_trainable_weights(self):
        model, x = dummy_model_and_inputs()
        model.compile(loss='mse', optimizer='adam')
        y = np.random.uniform(size=len(x))
        grad_trainable_weights = get_gradients_of_trainable_weights(model, x, y)

        self.assertListEqual(list(grad_trainable_weights), ['model/fc1/kernel:0', 'model/fc1/bias:0',
                                                            'block/fc1/kernel:0',
                                                            'block/fc1/bias:0',
                                                            'fc1/kernel:0',
                                                            'fc1/bias:0'])
        w1 = grad_trainable_weights['block/fc1/kernel:0']
        b1 = grad_trainable_weights['block/fc1/bias:0']
        w2 = grad_trainable_weights['fc1/kernel:0']
        b2 = grad_trainable_weights['fc1/bias:0']

        self.assertListEqual(list(w1.shape), [10, 10])  # Dense.w
        self.assertListEqual(list(b1.shape), [10, ])  # Dense.w
        self.assertListEqual(list(w2.shape), [10, 1])  # Dense.w
        self.assertListEqual(list(b2.shape), [1, ])  # Dense.b

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

    def test_multi_inputs_multi_outputs(self):
        inp_a = np.random.uniform(size=(5, 10))
        inp_b = np.random.uniform(size=(5, 10))
        # out_d = np.random.uniform(size=(5, 1))
        # out_e = np.random.uniform(size=(5, 1))

        m1 = get_multi_outputs_model()
        m1.compile(optimizer='adam', loss='mse')
        # m1.fit(x=[inp_a, inp_b], y=[out_d, out_e])
        acts = keract.get_activations(m1, [inp_a, inp_b])
        self.assertListEqual(list(acts['i1'].shape), [5, 10])
        self.assertListEqual(list(acts['i2'].shape), [5, 10])
        self.assertListEqual(list(acts['add'].shape), [5, 10])
        self.assertListEqual(list(acts['o1'].shape), [5, 1])
        self.assertListEqual(list(acts['o2'].shape), [5, 2])

    def test_model_in_model(self):
        np.random.seed(123)
        inputs = np.random.uniform(size=(8, 32, 32, 1))
        nested_model = create_network_with_one_subnet()

        # will get the activations of every layer, EXCLUDING subnet layers.
        acts_not_nested = keract.get_activations(nested_model, inputs)
        self.assertTrue('subnet' in acts_not_nested)
        self.assertTrue('subnet/conv2d' not in acts_not_nested)

        # will get the activations of every layer, including subnet.
        acts_nested = keract.get_activations(nested_model, inputs, nested=True)
        self.assertTrue('subnet' not in acts_nested)
        self.assertTrue('subnet/conv2d' in acts_nested)
        self.assertTrue('subnet/max_pooling2d' in acts_nested)
        self.assertTrue('subnet/conv2d_1' in acts_nested)
        self.assertTrue('subnet/max_pooling2d_1' in acts_nested)

    def test_get_activations_from_multi_outputs(self):
        # define model
        inputs = tf.keras.Input(shape=(None,), name='input')
        emb = tf.keras.layers.Embedding(100, 4, name='embeddding')(inputs)
        lstm = tf.keras.layers.LSTM(4, return_sequences=True, return_state=True, name='lstm')
        lstm_outputs, state_h, state_c = lstm(emb)
        outputs = tf.keras.layers.Dense(1, name='final_dense')(state_h)

        model = tf.keras.Model(inputs, outputs)
        model.summary()

        # create example
        x = np.ones(shape=(16, 10))

        # get activations
        act = keract.get_activations(model, x, layer_names='lstm')['lstm']
        self.assertEqual(len(act), 3)
        self.assertListEqual(list(act[0].shape), [16, 10, 4])
        self.assertListEqual(list(act[1].shape), [16, 4])
        self.assertListEqual(list(act[2].shape), [16, 4])
