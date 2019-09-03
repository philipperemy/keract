import numpy as np
from keras import Sequential
from keras.layers import Dropout, Dense

from keract import get_activations, persist_to_json_file, load_activations_from_json_file

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

# print them.
print(list(activations.keys()))
print(list(activations2.keys()))
