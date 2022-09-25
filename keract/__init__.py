import importlib

tf_spec = importlib.util.find_spec("tensorflow")
if tf_spec is None:
    raise ImportError("No valid tensorflow installation found. Please install "
                      "tensorflow>=2.0 or tensorflow-gpu>=2.0")

from keract.keract import display_activations  # noqa
from keract.keract import display_gradients_of_trainable_weights  # noqa
from keract.keract import display_heatmaps  # noqa
from keract.keract import get_activations  # noqa
from keract.keract import get_gradients_of_activations  # noqa
from keract.keract import get_gradients_of_trainable_weights  # noqa
from keract.keract import load_activations_from_json_file  # noqa
from keract.keract import persist_to_json_file  # noqa

__version__ = '4.5.1'
