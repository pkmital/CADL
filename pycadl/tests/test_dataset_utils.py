import numpy as np
import tensorflow as tf
from cadl import dataset_utils as dsu
from cadl.utils import exists
from numpy.testing import run_module_suite


class TestDatasetUtils:

    def test_gtzan_exists(self):
        assert (exists('http://opihi.cs.uvic.ca/sound/music_speech.tar.gz'))

    def test_gtzan_loads(self):
        Xs, Ys = dsu.gtzan_music_speech_load()
        assert (Xs.shape == (128, 2583, 256, 2))
        assert (Ys.shape == (128,))
        assert (np.sum(Ys) == 64)

    def test_cifar_exists(self):
        assert (
            exists('http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'))

    def test_cifar_loads(self):
        Xs, Ys = dsu.cifar10_load()
        assert (Xs.shape == (50000, 32, 32, 3))
        assert (Ys.shape == (50000,))
        assert (np.mean(Ys) == 4.5)
        assert (np.mean(Xs) == 120.70756512369792)

    def test_tiny_imagenet_exists(self):
        assert ('http://cs231n.stanford.edu/tiny-imagenet-200.zip')

    def test_tiny_imagenet_loads(self):
        Xs, Ys = dsu.tiny_imagenet_load()
        assert (len(Xs) == 120000)
        assert (Xs[0].endswith('n02226429_40.JPEG'))
        assert (Ys[0] == 'grasshopper, hopper')

    def test_input_pipeline(self):
        Xs, Ys = dsu.tiny_imagenet_load()
        n_batches = 0
        batch_size = 10
        with tf.Graph().as_default(), tf.Session() as sess:
            batch_generator = dsu.create_input_pipeline(
                Xs[:100],
                batch_size=batch_size,
                n_epochs=1,
                shape=(64, 64, 3),
                crop_shape=(64, 64, 3))
            init_op = tf.group(tf.global_variables_initializer(),
                               tf.local_variables_initializer())
            sess.run(init_op)
            coord = tf.train.Coordinator()
            tf.get_default_graph().finalize()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            try:
                while not coord.should_stop():
                    batch = sess.run(batch_generator)
                    assert (batch.shape == (batch_size, 64, 64, 3))
                    n_batches += 1
            except tf.errors.OutOfRangeError:
                pass
            finally:
                coord.request_stop()
            coord.join(threads)
        assert (n_batches == 10)

    def test_dataset(self):
        Xs, Ys = dsu.cifar10_load()
        ds = dsu.Dataset(Xs=Xs, ys=Ys, split=[0.8, 0.1, 0.1], one_hot=False)
        assert (ds.X.shape == (50000, 32, 32, 3))
        assert (ds.Y.shape == (50000,))

    def test_dataset_split(self):
        Xs, Ys = dsu.cifar10_load()
        ds = dsu.Dataset(Xs=Xs, ys=Ys, split=[0.8, 0.1, 0.1], one_hot=False)
        assert (ds.train.images.shape == (40000, 32, 32, 3))
        assert (ds.valid.images.shape == (5000, 32, 32, 3))
        assert (ds.test.images.shape == (5000, 32, 32, 3))

    def test_dataset_split_batch_generator(self):
        Xs, Ys = dsu.cifar10_load()
        ds = dsu.Dataset(Xs=Xs, ys=Ys, split=[0.8, 0.1, 0.1], one_hot=False)
        X_i, Y_i = next(ds.train.next_batch())
        assert (X_i.shape == (100, 32, 32, 3))
        assert (Y_i.shape == (100,))

    def test_dataset_onehot(self):
        Xs, Ys = dsu.cifar10_load()
        ds = dsu.Dataset(
            Xs=Xs, ys=Ys, split=[0.8, 0.1, 0.1], one_hot=True, n_classes=10)
        X_i, Y_i = next(ds.train.next_batch())
        assert (X_i.shape == (100, 32, 32, 3))
        assert (Y_i.shape == (100, 10))


if __name__ == "__main__":
    run_module_suite()
