'''Trains a simple convnet on the MNIST dataset.

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function

import keras
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential

from data import get_mnist_data, num_classes, input_shape

if __name__ == '__main__':

    x_train, y_train, x_test, y_test = get_mnist_data()

    batch_size = 128
    epochs = 12

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    # Change starts here
    import shutil
    import os

    # delete folder and its content and creates a new one.
    try:
        shutil.rmtree('checkpoints')
    except:
        pass
    os.mkdir('checkpoints')

    checkpoint = ModelCheckpoint(monitor='val_acc',
                                 filepath='checkpoints/model_{epoch:02d}_{val_acc:.3f}.h5',
                                 save_best_only=True)

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test),
              callbacks=[checkpoint])

    # Change finishes here

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
