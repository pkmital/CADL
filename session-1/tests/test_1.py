import matplotlib
matplotlib.use('Agg')
from libs import utils
import numpy as np


def test_celeb():
    files = utils.get_celeb_files()
    assert(len(files) == 100)

def test_crop():
    assert(utils.imcrop_tosquare(np.zeros((64, 23))).shape == (23, 23))
    assert(utils.imcrop_tosquare(np.zeros((23, 53))).shape == (23, 23))
    assert(utils.imcrop_tosquare(np.zeros((23, 23))).shape == (23, 23))
    assert(utils.imcrop_tosquare(np.zeros((24, 53))).shape == (24, 24))

def test_montage():
    assert(utils.montage(np.zeros((100, 32, 32, 3))).shape == (331, 331, 3))

