from cadl import fastwavenet
from numpy.testing import run_module_suite


def test_fastwavenet_model():
    fastwavenet.create_generation_model()


def test_sequence_length():
    res = fastwavenet.get_sequence_length(n_stages=3, n_layers_per_stage=10)
    assert(res == 6144)


if __name__ == "__main__":
    run_module_suite()
