import tensorflow as tf


def print_names_and_shapes(activations: dict):
    for layer_name, layer_activations in activations.items():
        print(layer_name)
        print(layer_activations.shape)
        print('')
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
    if tf.test.is_gpu_available():
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            for dev in physical_devices:
                tf.config.experimental.set_memory_growth(dev, True)
