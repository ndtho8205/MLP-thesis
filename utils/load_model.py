"""Load pickled model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle as pkl


def load(path):
    """Read a pickled model from a string.
    """
    with open(path, 'rb') as file:
        model = pkl.load(file)
    return model
