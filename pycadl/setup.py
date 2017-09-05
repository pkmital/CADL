import codecs
import os
import re

from setuptools import setup
from setuptools import find_packages

here = os.path.abspath(os.path.dirname(__file__))


def read(*parts):
    # intentionally *not* adding an encoding option to open
    # see here: https://github.com/pypa/virtualenv/issues/201#issuecomment-3145690
    return codecs.open(os.path.join(here, *parts), 'r').read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")

with open('README.md') as readme_file:
    readme = readme_file.read()

with open('HISTORY.md') as history_file:
    history = history_file.read()

requirements = [
    'numpy',
    'requests',
    'tensorflow',
    'scipy',
    'matplotlib',
    'scikit-image',
    'magenta',
    'nltk',
    'librosa',
    'bs4',
    'scikit-learn'
]


setup(
    name='cadl',
    version=find_version("cadl", "__init__.py"),
    description="Creative Applications of Deep Learning with TensorFlow",
    long_description=readme + '\n\n' + history,
    author="Parag Mital",
    author_email='parag@pkmital.com',
    url='https://github.com/pkmital/cadl/pycadl',
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
    classifiers=[
        'Development Status :: 3 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6'
    ],
    tests_require=['tox']
)


