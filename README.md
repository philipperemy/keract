# Keras Activations
```
pip install keract
```
*You have just found a (easy) way to get the activations for each layer of your Keras model (recurrent nets, conv nets, resnets...).*

<p align="center">
  <img src="assets/1.png">
</p>


## API

```
from keract import get_activations
get_activations(model, x)
```

### Inputs
- `model` is a `keras.models.Model` object
- `x` is a numpy array to feed to the model as input. In the case of multi-input, `x` is of type List. We use the Keras convention (as used in predict, fit...).

### Output
- A dictionary containing the activations for each layer of `model` for the input `x`:

```
{
  'conv2d_1/Relu:0': np.array(...),
  'conv2d_2/Relu:0': np.array(...),
  ...,
  'dense_2/Softmax:0': np.array(...)
}
```

The key is the name of the layer and the value is the corresponding output of the layer for the given input `x`.

## Examples

Examples are provided for:
- `keras.models.Sequential` - mnist.py
- `keras.models.Model` - multi_inputs.py
- Recurrent networks - recurrent.py

In the case of MNIST with LeNet, we are able to fetch the activations for a batch of size 128:

```
conv2d_1/Relu:0
(128, 26, 26, 32)

conv2d_2/Relu:0
(128, 24, 24, 64)

max_pooling2d_1/MaxPool:0
(128, 12, 12, 64)

dropout_1/cond/Merge:0
(128, 12, 12, 64)

flatten_1/Reshape:0
(128, 9216)

dense_1/Relu:0
(128, 128)

dropout_2/cond/Merge:0
(128, 128)

dense_2/Softmax:0
(128, 10)
```

We can even visualise some of them.

<p align="center">
  <img src="assets/0.png" width="50">
  <br><i>A random seven from MNIST</i>
</p>


<p align="center">
  <img src="assets/1.png">
  <br><i>Activation map of CONV1 of LeNet</i>
</p>

<p align="center">
  <img src="assets/2.png" width="200">
  <br><i>Activation map of FC1 of LeNet</i>
</p>


<p align="center">
  <img src="assets/3.png" width="300">
  <br><i>Activation map of Softmax of LeNet. <b>Yes it's a seven!</b></i>
</p>

