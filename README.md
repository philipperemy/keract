# Visualize the Activations of your layers with Keras
*Simple example to show how to get the activations for each layer in your Keras model*

This is the function to visualize the activations:
```
import keras.backend as K
import numpy as np


def get_visualizations(model, inputs, print_shape_only=False):
    print('----- activations -----')
    activations = []
    inp = model.input
    outputs = [layer.output for layer in model.layers]  # all layer outputs
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
```


Inputs:
- `model`: Keras model
- `inputs`: Inputs to the model for which we want to get the activations (for example 200 MNIST digits)
- `print_shape_only`: If set to True, will print the entire activations arrays (might be very verbose!)


I also provide a simple example to see how it works with the MNIST model. I separated the training and the visualizations because if the two are done sequentially, we have to re-train the model every time we want to visualize the activations! Not very practical! Here are the main steps:

## 1. Train your favorite model (I chose MNIST)
```
python model_train.py
```
- define the model
- train the model
- save the best model in checkpoints/

## 2. Visualize the activations of each layer
```
python read_activations.py
```
- load the model from the best checkpoint
- read the activations

### Examples
Shapes of the activations (one sample):
```
----- activations -----
(1, 26, 26, 32)
(1, 24, 24, 64)
(1, 12, 12, 64)
(1, 12, 12, 64)
(1, 9216)
(1, 128)
(1, 128)
(1, 10) # output of the softmax!
```

Shapes of the activations (200 samples):
```
----- activations -----
(200, 26, 26, 32)
(200, 24, 24, 64)
(200, 12, 12, 64)
(200, 12, 12, 64)
(200, 9216)
(200, 128)
(200, 128)
(200, 10)
```
