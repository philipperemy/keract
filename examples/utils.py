def print_names_and_shapes(activations):  # dict
    for layer_name, layer_activations in activations.items():
        print(layer_name)
        print(layer_activations.shape)
        print('')


def print_names_and_values(activations):  # dict
    for layer_name, layer_activations in activations.items():
        print(layer_name)
        print(layer_activations)
        print('')
