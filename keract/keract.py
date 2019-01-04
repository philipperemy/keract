import keras.backend as K
from keras.models import Model


def _evaluate(model: Model, nodes_to_evaluate, x, y=None):
    symb_inputs = (model._feed_inputs + model._feed_targets + model._feed_sample_weights)
    f = K.function(symb_inputs, nodes_to_evaluate)
    x_, y_, sample_weight_ = model._standardize_user_data(x, y)
    return f(x_ + y_ + sample_weight_)


# looks good.
def get_gradients_of_weights(model, model_inputs, outputs):
    if model.optimizer is None:
        raise Exception('Please compile your model first.')
    grads = model.optimizer.get_gradients(model.total_loss, model.trainable_weights)
    output_grad = _evaluate(model, grads, model_inputs, outputs)
    weight_names = [w.name for w in model.trainable_weights]
    result = dict(zip(weight_names, output_grad))
    return result


def get_gradients_of_activations(model, model_inputs, outputs, layer_name=None):
    if model.optimizer is None:
        raise Exception('Please compile your model first.')
    """ Gets gradient a layer output for given inputs and outputs"""
    # grads = model.optimizer.get_gradients(model.total_loss, model.layers[layer].output)
    layer_names = [l.output.name for l in model.layers]
    grads = model.optimizer.get_gradients(model.total_loss, [l.output for l in model.layers])
    output_grad = _evaluate(model, grads, model_inputs, outputs)
    result = dict(zip(layer_names, output_grad))
    return result


def get_activations(model, model_inputs, layer_name=None):
    outputs = [layer.output for layer in model.layers if layer.name == layer_name or layer_name is None]

    # we process the placeholders later (Inputs node in Keras). Because there's a bug in Tensorflow.
    input_layer_outputs = []
    layer_outputs = []
    for output in outputs:
        if 'input_' in output.name:
            input_layer_outputs.append(output)
        else:
            layer_outputs.append(output)

    activations = _evaluate(model, layer_outputs, model_inputs, None)
    names = [output.name for output in layer_outputs]
    result = dict(zip(names, activations))

    # add back the input layers.
    input_names = [output.name for output in input_layer_outputs]
    input_result = dict(zip(input_names, model_inputs))

    z = input_result.copy()
    z.update(result)
    return z


def display_activations(activations):
    import numpy as np
    import matplotlib.pyplot as plt
    """
    (1, 26, 26, 32)
    (1, 24, 24, 64)
    (1, 12, 12, 64)
    (1, 12, 12, 64)
    (1, 9216)
    (1, 128)
    (1, 128)
    (1, 10)
    """
    layer_names = list(activations.keys())
    activation_maps = list(activations.values())
    batch_size = activation_maps[0].shape[0]
    assert batch_size == 1, 'One image at a time to visualize.'
    for i, activation_map in enumerate(activation_maps):
        print('Displaying activation map {}'.format(i))
        shape = activation_map.shape
        if len(shape) == 4:
            activations = np.hstack(np.transpose(activation_map[0], (2, 0, 1)))
        elif len(shape) == 2:
            # try to make it square as much as possible. we can skip some activations.
            activations = activation_map[0]
            num_activations = len(activations)
            if num_activations > 1024:  # too hard to display it on the screen.
                square_param = int(np.floor(np.sqrt(num_activations)))
                activations = activations[0: square_param * square_param]
                activations = np.reshape(activations, (square_param, square_param))
            else:
                activations = np.expand_dims(activations, axis=0)
        else:
            raise Exception('len(shape) = 3 has not been implemented.')
        plt.title(layer_names[i])
        plt.imshow(activations, interpolation='None', cmap='jet')
        plt.show()
