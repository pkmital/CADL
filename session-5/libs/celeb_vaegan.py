"""
Creative Applications of Deep Learning w/ Tensorflow.
Kadenze, Inc.
Copyright Parag K. Mital, June 2016.
"""
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
from .utils import download
from skimage.transform import resize as imresize


def celeb_vaegan_download():
    """Download a pretrained inception network.

    Parameters
    ----------
    data_dir : str, optional
        Location of the pretrained inception network download.
    version : str, optional
        Version of the model: ['v3'] or 'v5'.
    """
    # For some reason, models w/ batch norm aren't working w/ tensorflow
    # right now.  A few bug tickets still open for this.  We'll have to
    # use the trainable checkpoint instead.
    # path1 = download('https://s3.amazonaws.com/cadl/models/celeb_vaegan.tfmodel')

    # Load the checkpoint
    model = download('https://s3.amazonaws.com/cadl/models/celeb.vaegan.tfmodel')
    labels = download('https://s3.amazonaws.com/cadl/celeb-align/list_attr_celeba.txt')
    return model, labels


def get_celeb_vaegan_model():
    """Get a pretrained model  network.

    Parameters
    ----------
    data_dir : str, optional
        Location of the pretrained inception network download.
    version : str, optional
        Version of the model: ['v3'] or 'v5'.

    Returns
    -------
    net : dict
        {'graph_def': graph_def, 'labels': synsets}
        where the graph_def is a tf.GraphDef and the synsets
        map an integer label from 0-1000 to a list of names
    """
    # Download the trained net
    model, labels = celeb_vaegan_download()

    # Parse the ids and synsets
    txt = open(labels).readlines()
    n_els = int(txt[0].strip())
    labels = txt[1].strip().split()
    n_labels = len(labels)
    attributes = np.zeros((n_els, n_labels), dtype=bool)
    for i, txt_i in enumerate(txt[2:]):
        attributes[i] = (np.array(txt_i.strip().split()[1:]).astype(int) > 0)

    # Load the saved graph
    with gfile.GFile(model, 'rb') as f:
        graph_def = tf.GraphDef()
        try:
            graph_def.ParseFromString(f.read())
        except:
            print('try adding PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python' +
                  'to environment.  e.g.:\n' +
                  'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python ipython\n' +
                  'See here for info: ' +
                  'https://github.com/tensorflow/tensorflow/issues/582')
    net = {
        'graph_def': graph_def,
        'labels': labels,
        'attributes': attributes,
        'preprocess': preprocess,
    }
    return net


def preprocess(img, crop_factor=0.8):
    """Replicate the preprocessing we did on the VAE/GAN.

    This model used a crop_factor of 0.8 and crop size of [100, 100, 3].
    """
    crop = np.min(img.shape[:2])
    r = (img.shape[0] - crop) // 2
    c = (img.shape[1] - crop) // 2
    cropped = img[r: r + crop, c: c + crop]
    r, c, *d = cropped.shape
    if crop_factor < 1.0:
        amt = (1 - crop_factor) / 2
        h, w = int(c * amt), int(r * amt)
        cropped = cropped[h:-h, w:-w]
    rsz = imresize(cropped, (100, 100), preserve_range=False)
    return rsz
