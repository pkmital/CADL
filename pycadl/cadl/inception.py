"""Inception model, download, and preprocessing.
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
import os
import numpy as np
from tensorflow.python.platform import gfile
import tensorflow as tf
import matplotlib.pyplot as plt
from skimage.transform import resize as imresize
from .utils import download_and_extract_tar, download_and_extract_zip


def inception_download(data_dir='inception', version='v5'):
    """Download a pretrained inception network.

    Parameters
    ----------
    data_dir : str, optional
        Location of the pretrained inception network download.
    version : str, optional
        Version of the model: ['v3'] or 'v5'.

    Returns
    -------
    TYPE
        Description
    """
    if version == 'v3':
        download_and_extract_tar(
            'https://s3.amazonaws.com/cadl/models/inception-2015-12-05.tgz',
            data_dir)
        return (os.path.join(data_dir, 'classify_image_graph_def.pb'),
                os.path.join(data_dir, 'imagenet_synset_to_human_label_map.txt'))
    else:
        download_and_extract_zip(
            'https://s3.amazonaws.com/cadl/models/inception5h.zip', data_dir)
        return (os.path.join(data_dir, 'tensorflow_inception_graph.pb'),
                os.path.join(data_dir, 'imagenet_comp_graph_label_strings.txt'))


def get_inception_model(data_dir='inception', version='v5'):
    """Get a pretrained inception network.

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
    model, labels = inception_download(data_dir, version)

    # Parse the ids and synsets
    txt = open(labels).readlines()
    synsets = [(key, val.strip()) for key, val in enumerate(txt)]

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
    return {
        'graph_def': graph_def,
        'labels': synsets,
        'preprocess': preprocess,
        'deprocess': deprocess
    }


def preprocess(img, crop=True, resize=True, dsize=(299, 299)):
    """Summary

    Parameters
    ----------
    img : TYPE
        Description
    crop : bool, optional
        Description
    resize : bool, optional
        Description
    dsize : tuple, optional
        Description

    Returns
    -------
    TYPE
        Description
    """
    if img.dtype != np.uint8:
        img *= 255.0

    if crop:
        crop = np.min(img.shape[:2])
        r = (img.shape[0] - crop) // 2
        c = (img.shape[1] - crop) // 2
        cropped = img[r: r + crop, c: c + crop]
    else:
        cropped = img

    if resize:
        rsz = imresize(cropped, dsize, preserve_range=True)
    else:
        rsz = cropped

    if rsz.ndim == 2:
        rsz = rsz[..., np.newaxis]

    rsz = rsz.astype(np.float32)
    # subtract imagenet mean
    return (rsz - 117)


def deprocess(img):
    """Summary

    Parameters
    ----------
    img : TYPE
        Description

    Returns
    -------
    TYPE
        Description
    """
    return np.clip(img + 117, 0, 255).astype(np.uint8)


def test_inception():
    """Loads the inception network and applies it to a test image.
    """
    with tf.Session() as sess:
        net = get_inception_model()
        tf.import_graph_def(net['graph_def'], name='inception')
        g = tf.get_default_graph()
        names = [op.name for op in g.get_operations()]
        x = g.get_tensor_by_name(names[0] + ':0')
        softmax = g.get_tensor_by_name(names[-3] + ':0')

        from skimage import data
        img = preprocess(data.coffee())[np.newaxis]
        res = np.squeeze(softmax.eval(feed_dict={x: img}))
        print([(res[idx], net['labels'][idx])
               for idx in res.argsort()[-5:][::-1]])

        """Let's visualize the network's gradient activation
        when backpropagated to the original input image.  This
        is effectively telling us which pixels contribute to the
        predicted class or given neuron"""
        pools = [name for name in names if 'pool' in name.split('/')[-1]]
        fig, axs = plt.subplots(1, len(pools))
        for pool_i, poolname in enumerate(pools):
            pool = g.get_tensor_by_name(poolname + ':0')
            pool.get_shape()
            neuron = tf.reduce_max(pool, 1)
            saliency = tf.gradients(neuron, x)
            neuron_idx = tf.arg_max(pool, 1)
            this_res = sess.run([saliency[0], neuron_idx],
                                feed_dict={x: img})

            grad = this_res[0][0] / np.max(np.abs(this_res[0]))
            axs[pool_i].imshow((grad * 128 + 128).astype(np.uint8))
            axs[pool_i].set_title(poolname)
