import tensorflow as tf
from cadl import charrnn
from cadl.utils import stdout_redirect
from numpy.testing import run_module_suite
import io


class TestCharRNN:
    def test_alice_char_rnn_model(self):
        with tf.Graph().as_default():
            charrnn.test_alice()

    def test_trump_char_rnn_model(self):
        with stdout_redirect(io.StringIO()) as stdout:
            with tf.Graph().as_default():
                charrnn.test_trump()
        stdout.seek(0)
        #stdout.readlines()[-1] == "Done training -- epoch limit reached\n"

if __name__ == "__main__":
    run_module_suite()
