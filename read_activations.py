from glob import glob

import keras.backend as K
import numpy as np
from keras.models import load_model
from natsort import natsorted

from data import get_mnist_data


def get_visualizations(m, inputs, print_shape_only=False):
    print('----- activations -----')
    activations = []
    inp = m.input
    outputs = [layer.output for layer in m.layers]  # all layer outputs
    funcs = [K.function([inp] + [K.learning_phase()], [out]) for out in outputs]  # evaluation functions
    if len(inputs.shape) == 3:
        batch_inputs = inputs[np.newaxis, ...]
    else:
        batch_inputs = inputs
    layer_outputs = [func([batch_inputs, 1.])[0] for func in funcs]
    for layer_activations in layer_outputs:
        activations.append(layer_activations)
        if print_shape_only:
            print(layer_activations.shape)
        else:
            print(layer_activations)
    return activations


if __name__ == '__main__':
    checkpoints = natsorted(glob('checkpoints/*.h5'))
    assert len(checkpoints) != 0, 'No checkpoints found.'
    checkpoint_file = checkpoints[-1]
    print('Loading [{}]'.format(checkpoint_file))
    model = load_model(checkpoint_file)

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    print(model.summary())

    x_train, y_train, x_test, y_test = get_mnist_data()

    batch_size = 128
    # checking that the accuracy is the same as before 99% at the first epoch.
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1, batch_size=128)
    print('')
    assert test_acc > 0.98

    get_visualizations(model, x_test[0], print_shape_only=True)  # with just one sample.

    get_visualizations(model, x_test[0:200], print_shape_only=True)  # with 200 samples.
