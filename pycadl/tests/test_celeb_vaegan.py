import numpy as np
import tensorflow as tf
from cadl import vaegan, celeb_vaegan, utils
from cadl.utils import exists, stdout_redirect
import io
from numpy.testing import run_module_suite



class TestCelebVAEGAN:
    def test_celeb_vaegan_model_exists(self):
        assert(exists('https://s3.amazonaws.com/cadl/models/celeb.vaegan.tfmodel'))

    def test_celeb_vaegan_labels_exists(self):
        assert(exists('https://s3.amazonaws.com/cadl/models/list_attr_celeba.txt'))

    def test_celeb_files_exist(self):
        assert(len(utils.get_celeb_files()) == 100)

    def test_celeb_vaegan_training(self):
        with stdout_redirect(io.StringIO()) as stdout:
            with tf.Graph().as_default():
                vaegan.test_celeb(n_epochs=1)
        stdout.seek(0)
        stdout.readlines()[-1] == "Done training -- epoch limit reached\n"

    def test_celeb_vaegan_model(self):
        with tf.Graph().as_default() as g, tf.Session(graph=g):
            net = celeb_vaegan.get_celeb_vaegan_model()
            tf.import_graph_def(
                net['graph_def'],
                name='net',
                input_map={
                    'encoder/variational/random_normal:0':
                    np.zeros(512, dtype=np.float32)
                }
            )
            names = [op.name for op in g.get_operations()]
            assert('net/x' in names)
            assert(len(names) == 168)

if __name__ == "__main__":
    run_module_suite()
