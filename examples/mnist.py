from __future__ import print_function

import os
from glob import glob

import keras
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential

import keract
import utils
from data import MNIST

# What this script does:
# - define the model
# - if no checkpoints are detected:
#   - train the model
#   - save the best model in checkpoints/
# - load the model from the best checkpoint
# - read the activations

if __name__ == '__main__':
    checkpoint_dir = 'checkpoints'

    checkpoints = glob(os.path.join(checkpoint_dir, '*.h5'))

    from keras.models import load_model

    if len(checkpoints) > 0:

        checkpoints = sorted(checkpoints)  # pip install natsort: natsorted() would be a better choice..
        assert len(checkpoints) != 0, 'No checkpoints found.'
        checkpoint_file = checkpoints[-1]
        print('Loading [{}]'.format(checkpoint_file))
        model = load_model(checkpoint_file)

        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        print(model.summary())

        x_train, y_train, x_test, y_test = MNIST.get_mnist_data()

        # checking that the accuracy is the same as before 99% at the first epoch.
        # test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0, batch_size=128)
        # print('')
        # assert test_acc > 0.98

        utils.print_names_and_shapes(keract.get_activations(model, x_test[0:200]))  # with 200 samples.
        utils.print_names_and_shapes(keract.get_gradients_of_trainable_weights(model, x_train[0:10], y_train[0:10]))
        utils.print_names_and_shapes(keract.get_gradients_of_activations(model, x_train[0:10], y_train[0:10]))

        a = keract.get_activations(model, x_test[0:1])  # with just one sample.
        keract.display_activations(a)

        # import numpy as np
        # import matplotlib.pyplot as plt
        # plt.imshow(np.squeeze(x_test[0:1]), interpolation='None', cmap='gray')
    else:
        x_train, y_train, x_test, y_test = MNIST.get_mnist_data()

        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3),
                         activation='relu',
                         input_shape=MNIST.input_shape))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(MNIST.num_classes, activation='softmax'))

        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])

        import shutil

        # delete folder and its content and creates a new one.
        if os.path.exists(checkpoint_dir):
            shutil.rmtree(checkpoint_dir)
        os.makedirs(checkpoint_dir)

        checkpoint = ModelCheckpoint(monitor='val_accuracy', save_best_only=True,
                                     filepath=os.path.join(checkpoint_dir, 'model_{epoch:02d}_{val_accuracy:.3f}.h5'))

        model.fit(x_train, y_train,
                  batch_size=128,
                  epochs=12,
                  verbose=1,
                  validation_data=(x_test, y_test),
                  callbacks=[checkpoint])

        score = model.evaluate(x_test, y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
