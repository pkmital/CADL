import matplotlib
matplotlib.use('Agg')
import tensorflow as tf
import numpy as np
from libs import utils
from libs import dataset_utils
from libs import charrnn
from libs import vaegan


def test_alice():
    charrnn.test_alice()


def test_trump():
    charrnn.test_trump()

