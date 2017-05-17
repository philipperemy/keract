# Keras Visualize Activations
*Simple example to show how to get the activations for each layer in your Keras model*

This example considers the MNIST model. I separated the training and the visualizations. If the two are done sequentially, we have to re-train the model every time we want to visualize the activations! Not very practical! Here are the main steps:

## 1. Train your favorite model (I chose MNIST)
```
python model_run.py
```
- definition of the model
- train the model
- save the best model in checkpoints/

## 2. Visualize the activations of each layer
```
python read_activations.py
```
- load the model from the best checkpoint
- read the activations:

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

## 3. Visualization function
`get_visualizations(model, inputs, print_shape_only=False) : activations`
Inputs:
- `model`: Keras model
- `inputs`: Inputs to the model for which we want to get the activations (for example 200 MNIST digits)
- `print_shape_only`: If set to True, will print the entire activations arrays (might be very verbose!)

