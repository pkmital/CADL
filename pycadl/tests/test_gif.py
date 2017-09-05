from cadl import gif
from cadl.datasets import CELEB
from numpy.testing import run_module_suite
from matplotlib.pyplot import imread
import os


def test_gif():
    files = CELEB()[:5]
    imgs = [imread(f) for f in files]
    gif.build_gif(imgs, saveto='test.gif')
    assert(os.path.exists('test.gif'))

if __name__ == "__main__":
    run_module_suite()
