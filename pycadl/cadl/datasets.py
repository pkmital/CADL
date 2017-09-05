"""Utils for loading common datasets.
"""
"""
Copyright 2017 Parag K. Mital.  See also NOTICE.md.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import tensorflow.examples.tutorials.mnist.input_data as input_data
import os
import numpy as np
from cadl.dataset_utils import \
    Dataset, cifar10_load, tiny_imagenet_load, gtzan_music_speech_load


def MNIST(one_hot=True, split=[1.0, 0.0, 0.0]):
    """Returns the MNIST dataset.

    Returns
    -------
    mnist : DataSet
        DataSet object w/ convenienve props for accessing
        train/validation/test sets and batches.

    Parameters
    ----------
    one_hot : bool, optional
        Description
    split : list, optional
        Description
    """
    ds = input_data.read_data_sets('MNIST_data/', one_hot=one_hot)
    return Dataset(
        np.r_[ds.train.images, ds.validation.images, ds.test.images],
        np.r_[ds.train.labels, ds.validation.labels, ds.test.labels],
        split=split)


def CIFAR10(flatten=True, split=[1.0, 0.0, 0.0]):
    """Returns the CIFAR10 dataset.

    Parameters
    ----------
    flatten : bool, optional
        Convert the 3 x 32 x 32 pixels to a single vector
    split : list, optional
        Description

    Returns
    -------
    cifar : Dataset
        Description
    """
    # plt.imshow(np.transpose(np.reshape(
    #   cifar.train.images[10], (3, 32, 32)), [1, 2, 0]))
    Xs, ys = cifar10_load()
    if flatten:
        Xs = Xs.reshape((Xs.shape[0], -1))
    return Dataset(Xs, ys, split=split)


def CELEB(path='./img_align_celeba/'):
    """Attempt to load the files of the CELEB dataset.

    Requires the files already be downloaded and placed in the `dst` directory.
    The first 100 files can be downloaded from the cadl.utils function get_celeb_files

    http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

    Parameters
    ----------
    path : str, optional
        Directory where the aligned/cropped celeb dataset can be found.

    Returns
    -------
    files : list
        List of file paths to the dataset.
    """
    if not os.path.exists(path):
        print('Could not find celeb dataset under {}.'.format(path))
        print('Try downloading the dataset from the "Aligned and Cropped" '
              'link located here (imgs/img_align_celeba.zip [1.34 GB]): '
              'http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html '
              'Or you can download the first 100 files using the '
              'utils.get_celeb_files function.')
        return None
    else:
        fs = [
            os.path.join(path, f) for f in os.listdir(path)
            if f.endswith('.jpg')
        ]
        if len(fs) < 202598:
            print(
                'WARNING: Loaded only a small subset of the CELEB dataset. '
                'If you want to use the entire 1.3 GB dataset, try downloading '
                'the dataset from the "Aligned and Cropped" '
                'link located here (imgs/img_align_celeba.zip [1.34 GB]): '
                'http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html')
        return fs


def TINYIMAGENET(path='./tiny_imagenet/'):
    """Attempt to load the files of the Tiny ImageNet dataset.

    http://cs231n.stanford.edu/tiny-imagenet-200.zip
    https://tiny-imagenet.herokuapp.com/

    Parameters
    ----------
    path : str, optional
        Directory where the dataset can be found or else will be placed.

    Returns
    -------
    files : list
        List of file paths to the dataset.
    labels : list
        List of labels for each file (only training files have labels)
    """
    files, labels = tiny_imagenet_load(path)
    return files, labels


def GTZAN(path='./gtzan_music_speech'):
    """Load the GTZAN Music and Speech dataset.

    Downloads the dataset if it does not exist into the dst directory.

    Parameters
    ----------
    path : str, optional
        Description

    Returns
    -------
    ds : Dataset
        Dataset object with array of data in X and array of labels in Y

    Deleted Parameters
    ------------------
    dst : str, optional
        Location of GTZAN Music and Speech dataset.
    """
    return Dataset(*gtzan_music_speech_load())
