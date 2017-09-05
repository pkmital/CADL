from numpy.testing import run_module_suite
from cadl import draw, dataset_utils as dsu


class TestDRAW:
    def test_draw_train_dataset(self):
        Xs, ys = dsu.cifar10_load()
        Xs = Xs[:100, ...].reshape((100, 3072))
        ys = ys[:100]
        ds = dsu.Dataset(Xs, ys, split=(0.5, 0.25, 0.25))
        draw.train_dataset(n_epochs=1, batch_size=25, ds=ds, A=32, B=32, C=3)

    def test_draw_train_input_pipeline(self):
        Xs, ys = dsu.tiny_imagenet_load()
        draw.train_input_pipeline(
                Xs[:100], batch_size=20, n_epochs=1, A=64, B=64, C=3,
                input_shape=(64, 64, 3))

if __name__ == "__main__":
    run_module_suite()
