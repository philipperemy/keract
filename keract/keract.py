import json
import os
from collections import OrderedDict

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow import Tensor
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model
from tensorflow.keras import Sequential


def _convert_1d_to_2d(num_units: int):
    # find divisors of num_units.
    divisors = []
    for i in range(1, num_units + 1):
        q = num_units / i
        if int(q) == q:
            divisors.append(i)
    divisors = list(reversed(divisors))
    pairs = []
    for d in divisors:
        for e in divisors[1:]:
            if d * e == num_units:
                pairs.append((d, e))
    if len(pairs) == 0:
        return num_units, 1
    # square x*y == rectangle x*y but minimizes x+y.
    close_to_square_id = int(np.argmin(np.sum(np.array(pairs), axis=1)))
    return pairs[close_to_square_id]


def n_(node, output_format_, nested=False):
    if isinstance(node, list):
        node_name = '_'.join([str(n.name) for n in node])
    else:
        node_name = str(node.name)
    if output_format_ == 'simple':
        if '/' in node_name:
            # This ensures that subnodes get properly named.
            tokens = node_name.split('/')
            if nested:
                return '/'.join(tokens[:-1])
            else:
                return tokens[0]
        elif ':' in node_name:
            return node_name.split(':')[0]
        else:
            return node_name
    return node_name


def _evaluate(model: Model, nodes_to_evaluate, x, y=None, auto_compile=False):
    if not model._is_compiled:
        # tensorflow.python.keras.applications.*
        applications_model_names = [
            'densenet',
            'efficientnet',
            'inception_resnet_v2',
            'inception_v3',
            'mobilenet',
            'mobilenet_v2',
            'nasnet',
            'resnet',
            'resnet_v2',
            'vgg16',
            'vgg19',
            'xception'
        ]
        if model.name in applications_model_names:
            print('Transfer learning detected. Model will be compiled with ("categorical_crossentropy", "adam").')
            print('If you want to change the default behaviour, then do in python:')
            print('model.name = ""')
            print('Then compile your model with whatever loss you want: https://keras.io/models/model/#compile.')
            print('If you want to get rid of this message, add this line before calling keract:')
            print('model.compile(loss="categorical_crossentropy", optimizer="adam")')
            model.compile(loss='categorical_crossentropy', optimizer='adam')
        else:
            if auto_compile:
                model.compile(loss='mse', optimizer='adam')
            else:
                print('Please compile your model first! https://keras.io/models/model/#compile.')
                print('If you only care about the activations (outputs of the layers), '
                      'then just compile your model like that:')
                print('model.compile(loss="mse", optimizer="adam")')
                raise Exception('Compilation of the model required.')

    def eval_fn(k_inputs):
        try:
            return K.function(k_inputs, nodes_to_evaluate)(model._standardize_user_data(x, y))
        except AttributeError:  # one way to avoid forcing non eager mode.
            if y is None:  # tf 2.3.0 upgrade compatibility.
                return K.function(k_inputs, nodes_to_evaluate)(x)
            return K.function(k_inputs, nodes_to_evaluate)((x, y))  # although works.
        except ValueError as e:
            print('Run it without eager mode. Paste those commands at the beginning of your script:')
            print('> import tensorflow as tf')
            print('> tf.compat.v1.disable_eager_execution()')
            raise e

    try:
        return eval_fn(model._feed_inputs + model._feed_targets + model._feed_sample_weights)
    except Exception:
        if tf.executing_eagerly():
            acts = []
            for n in list(nodes_to_evaluate):
                if isinstance(n, list):
                    acts.append([np.array(Model(model.inputs, nn)(x)) for nn in n])
                else:
                    if isinstance(n, Tensor):
                        acts.append(np.array(Model(model.inputs, n)(x)))
                    else:
                        acts.append(n.numpy())
            return acts
        return eval_fn(model._feed_inputs)


def get_gradients_of_trainable_weights(model, x, y):
    """
    Get the gradients of trainable_weights for the kernel and the bias nodes for all filters in each layer.
    Trainable_weights gradients are averaged over the minibatch.
    :param model: keras compiled model or one of ['vgg16', 'vgg19', 'inception_v3', 'inception_resnet_v2',
    'mobilenet_v2', 'mobilenetv2']
    :param x: inputs for which gradients are sought (averaged over all inputs if batch_size > 1)
    :param y: outputs for which gradients are sought
    :return: dict mapping layers to corresponding gradients (filter_h, filter_w, in_channels, out_channels)
    """
    nodes = OrderedDict([(n.name, n) for n in model.trainable_weights])
    return _get_gradients(model, x, y, nodes)


def get_gradients_of_activations(model, x, y, layer_names=None, output_format='simple', nested=False):
    """
    Get gradients of the outputs of the activation functions, regarding the loss.
    Intuitively, it shows how your activation maps change over a tiny modification of the loss.
    :param model: keras compiled model or one of ['vgg16', 'vgg19', 'inception_v3', 'inception_resnet_v2',
    'mobilenet_v2', 'mobilenetv2'].
    :param x: Model input (Numpy array). In the case of multi-inputs, x should be of type List.
    :param y: Model target (Numpy array). In the case of multi-inputs, y should be of type List.
    :param layer_names: (optional) Single name of a layer or list of layer names for which activations should be
    returned. It is useful in very big networks when it is computationally expensive to evaluate all the layers/nodes.
    :param output_format: Change the output dictionary key of the function.
    - 'simple': output key will match the names of the Keras layers. For example Dense(1, name='d1') will
    return {'d1': ...}.
    - 'full': output key will match the full name of the output layer name. In the example above, it will
    return {'d1/BiasAdd:0': ...}.
    - 'numbered': output key will be an index range, based on the order of definition of each layer within the model.
    :param nested: (optional) If set, will move recursively through the model definition to retrieve nested layers.
                Recursion ends at leaf layers of the model tree or at layers with their name specified in layer_names.

                E.g., a model with the following structure

                -layer1
                    -conv1
                    ...
                    -fc1
                -layer2
                    -fc2

                ... yields a dictionary with keys 'layer1/conv1', ..., 'layer1/fc1', 'layer2/fc2'.
                If layer_names = ['layer2/fc2'] is specified, the dictionary will only hold one key 'layer2/fc2'.

                The layer names are generated by joining all layers from top level to leaf level with the separator '/'.
    :return: Dict {layer_names (specified by output_format) -> activation of the layer output/node (Numpy array)}.
    """
    if tf.executing_eagerly():
        print('Run it without eager mode. Paste those commands at the beginning of your script:')
        print('> import tensorflow as tf')
        print('> tf.compat.v1.disable_eager_execution()')
        raise Exception('No eager mode for get_gradients_of_activations().')
    nodes = _get_nodes(model, output_format, nested=nested, layer_names=layer_names)
    return _get_gradients(model, x, y, nodes)


def _get_gradients(model, x, y, nodes):
    if model.optimizer is None:
        raise Exception('Please compile the model first. The loss function is required to compute the gradients.')

    nodes_names = nodes.keys()
    nodes_values = nodes.values()

    if tf.executing_eagerly():
        with tf.GradientTape(persistent=True) as tape:
            p = model(x)
            loss_fn = tf.keras.losses.get(model.loss) if isinstance(model.loss, str) else model.loss
            loss = loss_fn(p, y)
            # tape.gradient(tf.keras.losses.get(model.loss)(model(x), y), model.layers[2].output)
        gradients_values = tape.gradient(loss, list(nodes.values()))
    else:
        try:
            grads = model.optimizer.get_gradients(model.total_loss, nodes_values)
        except ValueError as e:
            if 'differentiable' in str(e):
                # Probably one of the gradients operations is not differentiable...
                grads = []
                differentiable_nodes = []
                for n in nodes_values:
                    try:
                        grads.extend(model.optimizer.get_gradients(model.total_loss, n))
                        differentiable_nodes.append(n)
                    except ValueError:
                        pass
                # nodes_values = differentiable_nodes
            else:
                raise e
        gradients_values = _evaluate(model, grads, x, y)
    return OrderedDict(zip(nodes_names, gradients_values))


def _get_nodes(module, output_format, nested=False, layer_names=[]):
    is_model_or_layer = isinstance(module, Model) or isinstance(module, Layer)
    has_layers = hasattr(module, '_layers') and bool(module._layers)
    assert is_model_or_layer, 'Not a model or layer!'

    def output(u):
        try:
            return u.output
        except AttributeError:  # for example Sequential. After tf2.3.
            return u.outbound_nodes[0].outputs

    try:
        module_name = n_(module.output, output_format_=output_format, nested=nested)
    except AttributeError:  # for example Sequential. After tf2.3.
        module_name = module.name

    if has_layers:
        node_dict = OrderedDict()
        # print('Layers:', module._layers)
        for m in module._layers:
            try:
                key = n_(m.output, output_format_=output_format, nested=nested)
            except AttributeError:  # for example Sequential. After tf2.3.
                try:
                    key = m.name
                except AttributeError:
                    return []

            if nested:
                nodes = _get_nodes(m, output_format,
                                   nested=nested,
                                   layer_names=layer_names)
            else:
                if bool(layer_names) and key in layer_names:
                    nodes = OrderedDict([(key, output(m))])
                elif not bool(layer_names):
                    nodes = OrderedDict([(key, output(m))])
                else:
                    nodes = OrderedDict()
            node_dict.update(nodes)
        return node_dict

    elif bool(layer_names) and module_name in layer_names:
        # print("1", module_name, module)
        return OrderedDict({module_name: module.output})

    elif not bool(layer_names):
        # print("2", module_name, module)
        return OrderedDict({module_name: module.output})

    else:
        # print("3", module_name, module)
        return OrderedDict()


def get_activations(model, x, layer_names=None, nodes_to_evaluate=None,
                    output_format='simple', nested=False, auto_compile=True):
    """
    Fetch activations (nodes/layers outputs as Numpy arrays) for a Keras model and an input X.
    By default, all the activations for all the layers are returned.
    :param model: Keras compiled model or one of ['vgg16', 'vgg19', 'inception_v3', 'inception_resnet_v2',
    'mobilenet_v2', 'mobilenetv2', ...].
    :param x: Model input (Numpy array). In the case of multi-inputs, x should be of type List.
    :param layer_names: (optional) Single name of a layer or list of layer names for which activations should be
    returned. It is useful in very big networks when it is computationally expensive to evaluate all the layers/nodes.
    :param nodes_to_evaluate: (optional) List of Keras nodes to be evaluated. Useful when the nodes are not
    in model.layers.
    :param output_format: Change the output dictionary key of the function.
    - 'simple': output key will match the names of the Keras layers. For example Dense(1, name='d1') will
    return {'d1': ...}.
    - 'full': output key will match the full name of the output layer name. In the example above, it will
    return {'d1/BiasAdd:0': ...}.
    - 'numbered': output key will be an index range, based on the order of definition of each layer within the model.
    :param nested: If specified, will move recursively through the model definition to retrieve nested layers.
                Recursion ends at leaf layers of the model tree or at layers with their name specified in layer_names.

                E.g., a model with the following structure

                -layer1
                    -conv1
                    ...
                    -fc1
                -layer2
                    -fc2

                ... yields a dictionary with keys 'layer1/conv1', ..., 'layer1/fc1', 'layer2/fc2'.
                If layer_names = ['layer2/fc2'] is specified, the dictionary will only hold one key 'layer2/fc2'.

                The layer names are generated by joining all layers from top level to leaf level with the separator '/'.
    :param auto_compile: If set to True, will auto-compile the model if needed.
    :return: Dict {layer_name (specified by output_format) -> activation of the layer output/node (Numpy array)}.
    """
    if tf.executing_eagerly() and any([isinstance(a, Sequential) for a in model.layers]):
        print('Run it without eager mode. Paste those commands at the beginning of your script:')
        print('> import tensorflow as tf')
        print('> tf.compat.v1.disable_eager_execution()')
        raise ValueError('Keract does not support eager mode and nested models (Sequential inside Sequential).')
    layer_names = [layer_names] if isinstance(layer_names, str) else layer_names
    # print('Layer names:', layer_names)
    if nodes_to_evaluate is None:
        nodes = _get_nodes(model, output_format, layer_names=layer_names, nested=nested)
    else:
        if layer_names is not None:
            raise ValueError('Do not specify a [layer_name] with [nodes_to_evaluate]. It will not be used.')
        nodes = OrderedDict([(n_(node, 'full'), node) for node in nodes_to_evaluate])

    if len(nodes) == 0:
        if layer_names is not None:
            network_layers = ', '.join([layer.name for layer in model.layers])
            raise KeyError('Could not find a layer with name: [{}]. '
                           'Network layers are [{}]'.format(', '.join(layer_names), network_layers))
        else:
            raise ValueError('Nodes list is empty. Or maybe the model is empty.')

    # The placeholders are processed later (Inputs node in Keras). Due to a small bug in tensorflow.
    input_layer_outputs = []
    layer_outputs = OrderedDict()

    for key, node in nodes.items():
        if isinstance(node, list):

            for nod in node:
                if tf.executing_eagerly():
                    layer_outputs.update({key: node})
                else:
                    if nod.op.type != 'Placeholder':  # no inputs please.
                        layer_outputs.update({key: node})
        else:
            if tf.executing_eagerly():
                layer_outputs.update({key: node})
            else:
                if node.op.type != 'Placeholder':  # no inputs please.
                    layer_outputs.update({key: node})
    if nodes_to_evaluate is None or (layer_names is not None) and \
            any([n.name in layer_names for n in model.inputs]):
        input_layer_outputs = list(model.inputs)

    if len(layer_outputs) > 0:
        activations = _evaluate(model, layer_outputs.values(), x, y=None, auto_compile=auto_compile)
    else:
        activations = {}

    def craft_output(output_format_):
        inputs = [x] if not isinstance(x, list) else x
        activations_inputs_dict = OrderedDict(
            zip([n_(output, output_format_) for output in input_layer_outputs], inputs))
        activations_dict = OrderedDict(zip(layer_outputs.keys(), activations))
        result_ = activations_inputs_dict.copy()
        result_.update(activations_dict)

        if output_format_ == 'numbered':
            result_ = OrderedDict([(i, v) for i, (k, v) in enumerate(result_.items())])
        return result_

    result = craft_output(output_format)
    if layer_names is not None:  # extra check.
        result = {k: v for k, v in result.items() if k in layer_names}
    if nodes_to_evaluate is not None and len(result) != len(nodes_to_evaluate):
        result = craft_output(output_format_='full')  # collision detected in the keys.

    return result


def display_activations(activations, cmap=None, save=False, directory='.',
                        data_format='channels_last', fig_size=(24, 24),
                        reshape_1d_layers=False):
    """
    Plot the activations for each layer using matplotlib
    :param activations: dict - mapping layers to corresponding activations (1, output_h, output_w, num_filters)
    :param cmap: string - a valid matplotlib colormap to be used
    :param save: bool - if the plot should be saved
    :param fig_size: (float, float), optional, default: None. width, height in inches.
    :param directory: string - where to store the activations (if save is True)
    :param data_format: string - one of "channels_last" (default) or "channels_first".
    :param reshape_1d_layers: tries to reshape large 1d layers to a square/rectangle.
    The ordering of the dimensions in the inputs. "channels_last" corresponds to inputs with
    shape (batch, steps, channels) (default format for temporal data in Keras) while "channels_first"
    corresponds to inputs with shape (batch, channels, steps).
    :return: None
    """
    import matplotlib.pyplot as plt
    import math
    index = 0
    for layer_name, acts in activations.items():
        print(layer_name, acts.shape, end=' ')
        if acts.shape[0] != 1:
            print('-> Skipped. First dimension is not 1.')
            continue

        print('')
        # channel first
        if data_format == 'channels_last':
            c = -1
        elif data_format == 'channels_first':
            c = 1
        else:
            raise Exception('Unknown data_format.')

        nrows = int(math.sqrt(acts.shape[c]) - 0.001) + 1  # best square fit for the given number
        ncols = int(math.ceil(acts.shape[c] / nrows))
        hmap = None
        if len(acts.shape) <= 2:
            """
            print('-> Skipped. 2D Activations.')
            continue
            """
            # no channel
            fig, axes = plt.subplots(1, 1, squeeze=False, figsize=fig_size)
            img = acts[0, :]
            img2 = np.reshape(img, _convert_1d_to_2d(img.shape[0])) if reshape_1d_layers else [img]
            hmap = axes.flat[0].imshow(img2, cmap=cmap)
            axes.flat[0].axis('off')
        else:
            fig, axes = plt.subplots(nrows, ncols, squeeze=False, figsize=fig_size)
            for i in range(nrows * ncols):
                if i < acts.shape[c]:
                    if len(acts.shape) == 3:
                        if data_format == 'channels_last':
                            img = acts[0, :, i]
                        elif data_format == 'channels_first':
                            img = acts[0, i, :]
                        else:
                            raise Exception('Unknown data_format.')
                        hmap = axes.flat[i].imshow([img], cmap=cmap)
                    elif len(acts.shape) == 4:
                        if data_format == 'channels_last':
                            img = acts[0, :, :, i]
                        elif data_format == 'channels_first':
                            img = acts[0, i, :, :]
                        else:
                            raise Exception('Unknown data_format.')
                        hmap = axes.flat[i].imshow(img, cmap=cmap)
                axes.flat[i].axis('off')
        fig.suptitle(layer_name)
        fig.subplots_adjust(right=0.8)
        cbar = fig.add_axes([0.85, 0.15, 0.03, 0.7])
        fig.colorbar(hmap, cax=cbar)
        if save:
            if not os.path.exists(directory):
                os.makedirs(directory)
            output_filename = os.path.join(directory, '{0}_{1}.png'.format(index, layer_name.split('/')[0]))
            plt.savefig(output_filename, bbox_inches='tight')
        else:
            plt.show()
        # pyplot figures require manual closing
        index += 1
        plt.close(fig)


def display_heatmaps(activations, input_image, directory='.', save=False, fix=True):
    """
    Plot heatmaps of activations for all filters overlayed on the input image for each layer
    :param activations: dict mapping layers to corresponding activations with the shape
    (1, output height, output width, number of filters)
    :param input_image: numpy array, input image for the overlay
    :param save: bool, if the plot should be saved
    :param fix: bool, if automated checks and fixes for incorrect images should be run
    :param directory: string - where to store the activations (if save is True)
    :return: None
    """
    from PIL import Image
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import MinMaxScaler
    import math

    data_format = K.image_data_format()
    if fix:
        # fixes common errors made when passing the image
        # I recommend the use of keras' load_img function passed to np.array to ensure
        # images are loaded in in the correct format
        # removes the batch size from the shape
        if len(input_image.shape) == 4:
            input_image = input_image.reshape(input_image.shape[1], input_image.shape[2], input_image.shape[3])
        # removes channels from the shape of grayscale images
        if len(input_image.shape) == 3 and input_image.shape[2] == 1:
            input_image = input_image.reshape(input_image.shape[0], input_image.shape[1])

    index = 0
    for layer_name, acts in activations.items():
        print(layer_name, acts.shape, end=' ')
        if acts.shape[0] != 1:
            print('-> Skipped. First dimension is not 1.')
            continue
        if len(acts.shape) <= 2:
            print('-> Skipped. 2D Activations.')
            continue
        print('')
        nrows = int(math.sqrt(acts.shape[-1]) - 0.001) + 1  # best square fit for the given number
        ncols = int(math.ceil(acts.shape[-1] / nrows))
        fig, axes = plt.subplots(nrows, ncols, figsize=(12, 12))
        fig.suptitle(layer_name)

        # computes values required to scale the activations (which will form our heat map) to be in range 0-1
        scaler = MinMaxScaler()
        # reshapes to be 2D with an automaticly calculated first dimension and second
        # dimension of 1 in order to keep scikitlearn happy
        scaler.fit(acts.reshape(-1, 1))

        # loops over each filter/neuron
        for i in range(nrows * ncols):
            if i < acts.shape[-1]:
                if len(acts.shape) == 3:
                    # gets the activation of the ith layer
                    if data_format == 'channels_last':
                        img = acts[0, :, i]
                    elif data_format == 'channels_first':
                        img = acts[0, i, :]
                    else:
                        raise Exception('Unknown data_format.')
                elif len(acts.shape) == 4:
                    if data_format == 'channels_last':
                        img = acts[0, :, :, i]
                    elif data_format == 'channels_first':
                        img = acts[0, i, :, :]
                    else:
                        raise Exception('Unknown data_format.')
                else:
                    raise Exception('Expect a tensor of 3 or 4 dimensions.')

                # scales the activation (which will form our heat map) to be in range 0-1 using
                # the previously calculated statistics
                if len(img.shape) == 1:
                    img = scaler.transform(img.reshape(-1, 1))
                else:
                    img = scaler.transform(img)
                # print(img.shape)
                img = Image.fromarray(img)
                # resizes the activation to be same dimensions of input_image
                img = img.resize((input_image.shape[1], input_image.shape[0]), Image.LANCZOS)
                img = np.array(img)
                if hasattr(axes, 'flat'):
                    axes.flat[i].imshow(input_image / 255.0)
                    # overlay the activation at 70% transparency  onto the image with a heatmap colour scheme
                    # Lowest activations are dark, highest are dark red, mid are yellow
                    axes.flat[i].imshow(img, alpha=0.3, cmap='jet', interpolation='bilinear')
                else:
                    axes.imshow(input_image / 255.0)
                    axes.imshow(img, alpha=0.3, cmap='jet', interpolation='bilinear')
            # axis off.
            axes.flat[i].axis('off') if hasattr(axes, 'flat') else axes.axis('off')

        if save:
            if not os.path.exists(directory):
                os.makedirs(directory)
            output_filename = os.path.join(directory, '{0}_{1}.png'.format(index, layer_name.split('/')[0]))
            plt.savefig(output_filename, bbox_inches='tight')
        else:
            plt.show()
        index += 1
        plt.close(fig)


def display_gradients_of_trainable_weights(gradients, directory='.', save=False):
    """
    Plot in_channels by out_channels grid of grad heatmaps each of dimensions (filter_h, filter_w)
    :param gradients: dict mapping layers to corresponding gradients (filter_h, filter_w, in_channels, out_channels)
    :param save: bool- if the plot should be saved
    :param directory: string - where to store the activations (if save is True)
    :return: None
    """
    import matplotlib.pyplot as plt

    index = 0
    for layer_name, grads in gradients.items():
        if len(grads.shape) != 4:
            print(layer_name, ": Expected dimensions (filter_h, filter_w, in_channels, out_channels). Got ",
                  grads.shape)
            continue
        print(layer_name, grads.shape)
        nrows = grads.shape[-1]
        ncols = grads.shape[-2]
        fig, axes = plt.subplots(nrows, ncols, figsize=(12, 12))
        fig.suptitle(layer_name)
        hmap = None
        for i in range(nrows):
            for j in range(ncols):
                g = grads[:, :, j, i]
                hmap = axes[i, j].imshow(g, aspect='auto')  # May cause distortion in case of in_out channel difference
                axes[i, j].axis('off')
        fig.subplots_adjust(right=0.8, wspace=0.02, hspace=0.3)
        cbar = fig.add_axes([0.85, 0.15, 0.03, 0.7])
        fig.colorbar(hmap, cax=cbar)
        if save:
            if not os.path.exists(directory):
                os.makedirs(directory)
            output_filename = os.path.join(directory, '{0}_{1}.png'.format(index, layer_name.split('/')[0]))
            plt.savefig(output_filename, bbox_inches='tight')
        else:
            plt.show()
        index += 1
        plt.close(fig)


def persist_to_json_file(activations, filename):
    """
    Persist the activations to the disk
    :param activations: activations (dict mapping layers)
    :param filename: output filename (JSON format)
    :return: None
    """
    with open(filename, 'w') as w:
        json.dump(fp=w, obj=OrderedDict({k: v.tolist() for k, v in activations.items()}), indent=2, sort_keys=False)


def load_activations_from_json_file(filename):
    """
    Read the activations from the disk
    :param filename: filename to read the activations from (JSON format)
    :return: activations (dict mapping layers)
    """
    with open(filename, 'r') as r:
        d = json.load(r, object_pairs_hook=OrderedDict)
        activations = OrderedDict({k: np.array(v) for k, v in d.items()})
        return activations
