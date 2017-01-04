import matplotlib
matplotlib.use('Agg')
import tensorflow as tf
import numpy as np
from libs import utils
from libs import dataset_utils
from libs import vae


def test_mnist_vae():
    vae.test_mnist(n_epochs=1)


def test_cifar():
    dataset_utils.cifar10_download('cifar10')

