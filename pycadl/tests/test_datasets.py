from cadl import datasets
from numpy.testing import run_module_suite


class TestDatasets:
    def test_mnist(self):
        ds = datasets.MNIST()
        assert(ds.X.shape == (70000, 784))

    def test_celeb(self):
        datasets.CELEB()

if __name__ == "__main__":
    run_module_suite()
