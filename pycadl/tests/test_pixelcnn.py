from cadl import pixelcnn
from cadl.utils import exists
from numpy.testing import run_module_suite


def test_pixelcnn_model():
    model = pixelcnn.build_conditional_pixel_cnn_model()
    assert('X' in model)
    assert('cost' in model)
    assert('preds' in model)
    assert('sampled_preds' in model)
    assert('summaries' in model)
    assert(model['X'].shape.as_list() == [None, 32, 32, 3])

if __name__ == "__main__":
    run_module_suite()
