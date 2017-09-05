from cadl import librispeech
from cadl.utils import exists
from numpy.testing import run_module_suite


def test_librispeech_exists():
    assert(exists('http://www.openslr.org/resources/12/dev-clean.tar.gz'))
    assert(exists('http://www.openslr.org/resources/12/train-clean-100.tar.gz'))
    assert(exists('http://www.openslr.org/resources/12/train-clean-360.tar.gz'))

def test_librispeech_dataset():
    ds = librispeech.get_dataset()
    assert(len(ds) == 106717)

def test_librispeech_batch():
    ds = librispeech.get_dataset()
    batch = next(librispeech.batch_generator(ds, batch_size=32))
    assert(batch[0].shape == (32, 6144))
    assert(batch[1].shape == (32,))

if __name__ == "__main__":
    run_module_suite()
