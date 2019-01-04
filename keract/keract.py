import keras.backend as K
from keras.models import Model


def _evaluate(model: Model, nodes_to_evaluate, x, y=None):
    symb_inputs = (model._feed_inputs + model._feed_targets + model._feed_sample_weights)
    f = K.function(symb_inputs, nodes_to_evaluate)
    x_, y_, sample_weight_ = model._standardize_user_data(x, y)
    return f(x_ + y_ + sample_weight_)


def get_gradients_of_trainable_weights(model, x, y):
    nodes = model.trainable_weights
    nodes_names = [w.name for w in nodes]
    return _get_gradients(model, x, y, nodes, nodes_names)


def get_gradients_of_activations(model, x, y, layer_name=None):
    nodes = [layer.output for layer in model.layers if layer.name == layer_name or layer_name is None]
    nodes_names = [n.name for n in nodes]
    return _get_gradients(model, x, y, nodes, nodes_names)


def _get_gradients(model, x, y, nodes, nodes_names):
    if model.optimizer is None:
        raise Exception('Please compile the model first. The loss function is required to compute the gradients.')
    grads = model.optimizer.get_gradients(model.total_loss, nodes)
    gradients_values = _evaluate(model, grads, x, y)
    result = dict(zip(nodes_names, gradients_values))
    return result


def get_activations(model, x, layer_name=None):
    nodes = [layer.output for layer in model.layers if layer.name == layer_name or layer_name is None]

    # we process the placeholders later (Inputs node in Keras). Because there's a bug in Tensorflow.
    input_layer_outputs = []
    layer_outputs = []
    for node in nodes:
        if 'input_' in node.name:
            input_layer_outputs.append(node)
        else:
            layer_outputs.append(node)

    activations = _evaluate(model, layer_outputs, x, y=None)

    activations_dict = dict(zip([output.name for output in layer_outputs], activations))
    activations_inputs_dict = dict(zip([output.name for output in input_layer_outputs], x))

    result = activations_inputs_dict.copy()
    result.update(activations_dict)
    return result


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
    for name, activation_map in activations.items():
        assert activation_map.shape[0] == 1, 'One image at a time to visualize.'
        print('Displaying activation map [{}]'.format(name))
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
        plt.title(name)
        plt.imshow(activations, interpolation='None', cmap='jet')
        plt.show()
