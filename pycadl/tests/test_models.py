import tensorflow as tf
from cadl import inception, vgg16, i2v
from cadl.utils import exists
from numpy.testing import run_module_suite

class TestModels():
    def test_inception_exists(self):
        assert(exists('https://s3.amazonaws.com/cadl/models/inception-2015-12-05.tgz'))

    def test_inception_loads(self):
        net = inception.get_inception_model()
        with tf.Graph().as_default() as g, tf.Session():
            tf.import_graph_def(net['graph_def'])
            ops = g.get_operations()
            names = [op.name for op in ops]
        assert(len(names) == 370)
        assert(names[-1].endswith('output2'))
        assert(names[0].endswith('input'))

    def test_i2v_exists(self):
        assert(exists('https://s3.amazonaws.com/cadl/models/illust2vec.tfmodel'))

    def test_i2v_loads(self):
        net = i2v.get_i2v_tag_model()
        with tf.Graph().as_default() as g, tf.Session():
            tf.import_graph_def(net['graph_def'])
            ops = g.get_operations()
            names = [op.name for op in ops]
        assert(len(names) == 115)
        assert(names[0].endswith('Placeholder'))
        assert(names[-1].endswith('prob'))

    def test_vgg16_exists(self):
        assert(exists('https://s3.amazonaws.com/cadl/models/vgg16.tfmodel'))

    def test_vgg16_loads(self):
        net = vgg16.get_vgg_model()
        with tf.Graph().as_default() as g, tf.Session():
            tf.import_graph_def(net['graph_def'])
            ops = g.get_operations()
            names = [op.name for op in ops]
        assert(len(names) == 129)
        assert(names[0].endswith('images'))
        assert(names[-1].endswith('init'))
        assert(names[-2].endswith('prob'))


if __name__ == "__main__":
    run_module_suite()
