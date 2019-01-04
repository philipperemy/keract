import keras.backend as K


# looks good.
def get_gradients_of_weights(model, model_inputs, outputs):
    if model.optimizer is None:
        raise Exception('Please compile your model first.')
    grads = model.optimizer.get_gradients(model.total_loss, model.trainable_weights)
    symb_inputs = (model._feed_inputs + model._feed_targets + model._feed_sample_weights)
    f = K.function(symb_inputs, grads)
    x, y, sample_weight = model._standardize_user_data(model_inputs, outputs)
    output_grad = f(x + y + sample_weight)
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
    symb_inputs = (model._feed_inputs + model._feed_targets + model._feed_sample_weights)
    f = K.function(symb_inputs, grads)
    x, y, sample_weight = model._standardize_user_data(model_inputs, outputs)
    output_grad = f(x + y + sample_weight)
    result = dict(zip(layer_names, output_grad))
    return result


def get_activations(model, model_inputs, layer_name=None):
    inp = model.input

    model_multi_inputs_cond = True
    if not isinstance(inp, list):
        # only one input! let's wrap it in a list.
        inp = [inp]
        model_multi_inputs_cond = False

    outputs = [layer.output for layer in model.layers if
               layer.name == layer_name or layer_name is None]  # all layer outputs

    # we remove the placeholders (Inputs node in Keras). Not the most elegant though..
    outputs = [output for output in outputs if 'input_' not in output.name]

    funcs = [K.function(inp + [K.learning_phase()], [out]) for out in outputs]  # evaluation functions

    if model_multi_inputs_cond:
        list_inputs = []
        list_inputs.extend(model_inputs)
        list_inputs.append(0.)
    else:
        list_inputs = [model_inputs, 0.]

    # Learning phase. 0 = Test mode (no dropout or batch normalization)
    # layer_outputs = [func([model_inputs, 0.])[0] for func in funcs]
    activations = [func(list_inputs)[0] for func in funcs]
    names = [output.name for output in outputs]

    result = dict(zip(names, activations))
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
