import numpy as np
from cadl import deepdream
from numpy.testing import run_module_suite


class TestDeepDream():
    def test_deepdream(self):
        img = np.random.rand(256, 256, 3)
        deepdream.deep_dream(img, n_iterations=5)

    def test_guided_deepdream(self):
        img = np.random.rand(256, 256, 3)
        deepdream.guided_dream(img, n_iterations=5)

if __name__ == "__main__":
    run_module_suite()
