import tensorflow as tf
from cadl import batch_norm
from numpy.testing import run_module_suite


class TestBatchNorm:
    def test_batch_norm(self):
        x = tf.placeholder(shape=[None, 1], dtype=tf.float32)
        phase_train = tf.placeholder(dtype=tf.bool)
        op = batch_norm.batch_norm(x=x, phase_train=phase_train)
        assert(op is not None)


if __name__ == "__main__":
    run_module_suite()
