import tensorflow as tf


def print_names_and_shapes(activations: dict):
    for layer_name, layer_activations in activations.items():
        if isinstance(layer_activations, float):
            print(layer_name, '<float>')
        else:
            print(layer_name, layer_activations.shape)
    print('-' * 80)


def print_names_and_values(activations: dict):
    for layer_name, layer_activations in activations.items():
        print(layer_name)
        print(layer_activations)
        print('')
    print('-' * 80)


def gpu_dynamic_mem_growth():
    # Check for GPUs and set them to dynamically grow memory as needed
    # Avoids OOM from tensorflow greedily allocating GPU memory
    try:
        gpu_devices = tf.config.list_physical_devices('GPU')
        if len(gpu_devices) > 0:
            for gpu in gpu_devices:
                tf.config.experimental.set_memory_growth(gpu, True)
    except AttributeError:
        print('Upgrade your tensorflow to 2.x to have the gpu_dynamic_mem_growth feature.')
