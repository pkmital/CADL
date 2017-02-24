import matplotlib
matplotlib.use('Agg')
import tensorflow as tf
import numpy as np
from libs import utils
from libs import dataset_utils
from libs import charrnn
from libs import vaegan
from libs import celeb_vaegan


def test_alice():
    charrnn.test_alice()


def test_trump():
    charrnn.test_trump()


def test_vaegan_training():
    utils.get_celeb_files()
    vaegan.test_celeb(1)


def test_celeb_vaegan():
    net = celeb_vaegan.get_celeb_vaegan_model()
    sess = tf.Session()
    g = tf.get_default_graph()
    tf.import_graph_def(
        net['graph_def'],
        name='net',
        input_map={
            'encoder/variational/random_normal:0':
            np.zeros(512, dtype=np.float32)
        }
    )
    names = [op.name for op in g.get_operations()]
    print(names)
