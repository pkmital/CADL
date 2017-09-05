from cadl import glove
from cadl.utils import exists
from numpy.testing import run_module_suite


def test_glove_exists():
    assert(exists('http://nlp.stanford.edu/data/wordvecs/glove.6B.zip'))

def test_glove_model():
    model = glove.get_model()
    assert(model[0].shape == (400001, 300))
    assert(len(model[1]) == 400001)

if __name__ == "__main__":
    run_module_suite()
