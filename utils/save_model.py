"""Save model using pickle."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle as pkl


def save(data, path):
    """Write a pickled representation of model to file.
    """
    with open(path, 'wb') as file:
        pkl.dump(data, file, pkl.HIGHEST_PROTOCOL)
