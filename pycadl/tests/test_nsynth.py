import numpy as np
from cadl import nsynth
from cadl.utils import exists, download
from numpy.testing import run_module_suite


def test_model_exists():
    assert(exists('https://s3.amazonaws.com/cadl/models/model.ckpt-200000.data-00000-of-00001'))
    assert(exists('https://s3.amazonaws.com/cadl/models/model.ckpt-200000.index'))
    assert(exists('https://s3.amazonaws.com/cadl/models/model.ckpt-200000.meta'))


def test_generation():
    download('https://s3.amazonaws.com/cadl/models/model.ckpt-200000.data-00000-of-00001')
    download('https://s3.amazonaws.com/cadl/models/model.ckpt-200000.index')
    download('https://s3.amazonaws.com/cadl/models/model.ckpt-200000.meta')
    wav_file = download('https://s3.amazonaws.com/cadl/share/trumpet.wav')
    res = nsynth.synthesize(wav_file, synth_length=100)
    max_idx = np.max(np.where(res))
    assert(max_idx > 90 and max_idx < 100)
    assert(np.max(res) > 0.0)


if __name__ == "__main__":
    run_module_suite()
