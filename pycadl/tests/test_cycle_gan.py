import numpy as np
from cadl import cycle_gan
from numpy.testing import run_module_suite


class TestCycleGAN:
    def test_training_dataset(self):
        img1 = np.random.rand(100, 256, 256, 3).astype(np.float32)
        img2 = np.random.rand(100, 256, 256, 3).astype(np.float32)
        cycle_gan.train(img1, img2, ckpt_path='test', n_epochs=1)


    def test_training_random_crop(self):
        img1 = np.random.rand(1024, 1024, 3).astype(np.float32)
        img2 = np.random.rand(1024, 1024, 3).astype(np.float32)
        cycle_gan.train(img1, img2, ckpt_path='test', n_epochs=1)

if __name__ == "__main__":
    run_module_suite()
