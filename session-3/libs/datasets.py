"""Creative Applications of Deep Learning w/ Tensorflow.
Kadenze, Inc.
Copyright Parag K. Mital, June 2016.
"""
import tensorflow.examples.tutorials.mnist.input_data as input_data
from .dataset_utils import *


def MNIST(one_hot=True, split=[1.0, 0.0, 0.0]):
    """Returns the MNIST dataset.

    Returns
    -------
    mnist : DataSet
        DataSet object w/ convenienve props for accessing
        train/validation/test sets and batches.
    """
    ds = input_data.read_data_sets('MNIST_data/', one_hot=one_hot)
    return Dataset(np.r_[ds.train.images,
                         ds.validation.images,
                         ds.test.images],
                   np.r_[ds.train.labels,
                         ds.validation.labels,
                         ds.test.labels],
                   split=split)


def CIFAR10(flatten=True, split=[1.0, 0.0, 0.0]):
    """Returns the CIFAR10 dataset.

    Parameters
    ----------
    flatten : bool, optional
        Convert the 3 x 32 x 32 pixels to a single vector

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
        print('Try downloading the dataset from the "Aligned and Cropped" ' +
              'link located here (imgs/img_align_celeba.zip [1.34 GB]): ' +
              'http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html')
        return None
    else:
        fs = [os.path.join(path, f)
              for f in os.listdir(path) if f.endswith('.jpg')]
        if len(fs) < 202598:
            print('It does not look like you have downloaded the entire ' +
                  'Celeb Dataset.\n' +
                  'Try downloading the dataset from the "Aligned and Cropped" ' +
                  'link located here (imgs/img_align_celeba.zip [1.34 GB]): ' +
                  'http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html')
        return fs
