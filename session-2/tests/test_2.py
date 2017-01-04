import matplotlib
matplotlib.use('Agg')
import tensorflow as tf
import numpy as np
from libs import utils


def test_flatten():
    assert(utils.flatten(
        tf.constant(np.zeros((3, 100, 100, 3)))).get_shape().as_list() == [3, 30000])


def test_linear():
    h, W = utils.linear(tf.constant(np.zeros((3, 100, 100, 3), dtype=np.float32)), 10)
    assert(h.get_shape().as_list() == [3, 10])


def test_montage():
    assert(utils.slice_montage(
        utils.montage(np.zeros((100, 3, 3, 3))), 3, 3, 100).shape == (100, 3, 3, 3))
