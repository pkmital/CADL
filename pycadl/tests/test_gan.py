import tensorflow as tf
from cadl import gan
from cadl.utils import stdout_redirect
from cadl.datasets import CELEB
from numpy.testing import run_module_suite
import io


def test_gan_model():
    with tf.Graph().as_default(), tf.Session():
        gan.GAN(input_shape=[None, 32, 32, 3], n_latent=10, n_features=16, rgb=True)

def test_gan_training():
    files = CELEB()[:100]
    with stdout_redirect(io.StringIO()) as stdout:
        with tf.Graph().as_default():
            gan.train_input_pipeline(files=files, n_epochs=1, batch_size=10)
    stdout.seek(0)
    stdout.readlines()[-1] == "Done training -- epoch limit reached\n"


if __name__ == "__main__":
    run_module_suite()
