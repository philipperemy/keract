from __future__ import print_function

from glob import glob

import keras.backend as K
import numpy as np
from scipy.misc import imresize

from data import get_mnist_data

# What this script does:
# - define the model
# - if no checkpoints are detected:
#   - train the model
#   - save the best model in checkpoints/
# - load the model from the best checkpoint
# - read the activations

if __name__ == '__main__':

    checkpoints = glob('examples/checkpoints/*.h5')
    # pip3 install natsort
    from natsort import natsorted

    from keras.models import load_model

    if len(checkpoints) > 0:

        checkpoints = natsorted(checkpoints)
        assert len(checkpoints) != 0, 'No checkpoints found.'
        checkpoint_file = checkpoints[-1]
        print('Loading [{}]'.format(checkpoint_file))
        model = load_model(checkpoint_file)

        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        print(model.summary())

        image_id = 33
        conv_output = model.layers[1].output
        x_train, y_train, x_test, y_test = get_mnist_data()
        conv_output_grads = K.gradients(model.total_loss, conv_output)[0]
        inputs = model._feed_inputs + model._feed_targets + model._feed_sample_weights + [K.learning_phase()]
        gradient_func = K.function(inputs, [conv_output_grads, conv_output])
        output_grad, output_activations = gradient_func([[x_train[image_id]], [y_train[image_id]], [1], False])
        mul_grad_act = np.sum(np.abs(output_grad * output_activations), axis=-1).squeeze()

        import matplotlib.pyplot as plt

        plt.imshow(x_train[image_id].squeeze())
        plt.show()

        plt.imshow(imresize(mul_grad_act, (28, 28)))
        plt.show()

    else:
        pass
