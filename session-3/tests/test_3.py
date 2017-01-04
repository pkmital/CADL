import matplotlib
matplotlib.use('Agg')
import tensorflow as tf
import numpy as np
from libs import utils
from libs import dataset_utils


def test_cifar():
    dataset_utils.cifar10_download('cifar10')

