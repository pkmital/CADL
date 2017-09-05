from cadl import cornell
from cadl.utils import exists
from numpy.testing import run_module_suite


class TestCornell:
    def test_cornell_exists(self):
        assert(exists('https://s3.amazonaws.com/cadl/models/cornell_movie_dialogs_corpus.zip'))

    def test_cornell_download(self):
        cornell.download_cornell()

    def test_cornell_scripts(self):
        scripts = cornell.get_scripts()
        assert(scripts[0] == 'Can we make this quick?  '
                'Roxanne Korrine and Andrew Barrett are having '
                'an incredibly horrendous public break- up on the quad.  Again.')
        assert(len(scripts) == 304713)

if __name__ == "__main__":
    run_module_suite()
