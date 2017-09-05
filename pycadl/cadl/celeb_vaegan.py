"""Tools for downloading the celeb dataset and model, including preprocessing.
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
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
from cadl.utils import download
from skimage.transform import resize as imresize


def celeb_vaegan_download():
    """Download a pretrained celeb vae/gan network.

    Returns
    -------
    TYPE
        Description
    """

    # Load the model and labels
    model = download(
        'https://s3.amazonaws.com/cadl/models/celeb.vaegan.tfmodel')
    labels = download(
        'https://s3.amazonaws.com/cadl/celeb-align/list_attr_celeba.txt')
    return model, labels


def get_celeb_vaegan_model():
    """Get a pretrained model.

    Returns
    -------
    net : dict
        {
            'graph_def': tf.GraphDef
                The graph definition
            'labels': list
                List of different possible attributes from celeb
            'attributes': np.ndarray
                One hot encoding of the attributes per image
                [n_els x n_labels]
            'preprocess': function
                Preprocess function
        }
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

    Parameters
    ----------
    img : TYPE
        Description
    crop_factor : float, optional
        Description

    Returns
    -------
    TYPE
        Description
    """
    crop = np.min(img.shape[:2])
    r = (img.shape[0] - crop) // 2
    c = (img.shape[1] - crop) // 2
    cropped = img[r:r + crop, c:c + crop]
    r, c, *d = cropped.shape
    if crop_factor < 1.0:
        amt = (1 - crop_factor) / 2
        h, w = int(c * amt), int(r * amt)
        cropped = cropped[h:-h, w:-w]
    rsz = imresize(cropped, (100, 100), preserve_range=False)
    return rsz
