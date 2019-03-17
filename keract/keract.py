import keras.backend as K
from keras.models import Model


def _evaluate(model: Model, nodes_to_evaluate, x, y=None):
    if not model._is_compiled:
        if model.name in ['vgg16', 'vgg19', 'inception_v3', 'inception_resnet_v2', 'mobilenet_v2', 'mobilenetv2']:
            print('Transfer learning detected. Model will be compiled with ("categorical_crossentropy", "adam").')
            print('If you want to change the default behaviour, then do in python:')
            print('model.name = ""')
            print('Then compile your model with whatever loss you want: https://keras.io/models/model/#compile.')
            print('If you want to get rid of this message, add this line before calling keract:')
            print('model.compile(loss="categorical_crossentropy", optimizer="adam")')
            model.compile(loss='categorical_crossentropy', optimizer='adam')
        else:
            print('Please compile your model first! https://keras.io/models/model/#compile.')
            print('If you only care about the activations (outputs of the layers), '
                  'then just compile your model like that:')
            print('model.compile(loss="mse", optimizer="adam")')
            raise Exception('Compilation of the model required.')
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
    input_layer_outputs, layer_outputs = [], []
    [input_layer_outputs.append(node) if 'input_' in node.name else layer_outputs.append(node) for node in nodes]
    activations = _evaluate(model, layer_outputs, x, y=None)
    activations_dict = dict(zip([output.name for output in layer_outputs], activations))
    activations_inputs_dict = dict(zip([output.name for output in input_layer_outputs], x))
    result = activations_inputs_dict.copy()
    result.update(activations_dict)
    return result


def display_activations(activations, save=False):
    import matplotlib.pyplot as plt
    import math
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
        for i in range(nrows * ncols):
            if i < acts.shape[-1]:
                img = acts[0, :, :, i]
                hmap = axes.flat[i].imshow(img)
            axes.flat[i].axis('off')
        fig.subplots_adjust(right=0.8)
        cbar = fig.add_axes([0.85, 0.15, 0.03, 0.7])
        fig.colorbar(hmap, cax=cbar)
        if save:
            plt.savefig(layer_name.split('/')[0] + '.png', bbox_inches='tight')
        else:
            plt.show()


def display_heatmaps(activations, image, save=False):
    from PIL import Image
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import MinMaxScaler
    import numpy as np
    import math
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
        scaler.fit(acts.reshape(-1, 1))
        for i in range(nrows * ncols):
            if i < acts.shape[-1]:
                img = acts[0, :, :, i]
                # scale the activations (which will form our heat map) to be in range 0-1
                img = scaler.transform(img)
                # resize heatmap to be same dimensions of image
                img = Image.fromarray(img)
                img = img.resize((image.shape[0], image.shape[1]), Image.BILINEAR)
                img = np.array(img)
                axes.flat[i].imshow(img / 255.0)
                # overlay a 70% transparent heat map onto the image
                # Lowest activations are dark, highest are dark red, mid are yellow
                axes.flat[i].imshow(img, alpha=0.3, cmap='jet', interpolation='bilinear')
            axes.flat[i].axis('off')
        if save:
            plt.savefig(layer_name.split('/')[0] + '.png', bbox_inches='tight')
        else:
            plt.show()
