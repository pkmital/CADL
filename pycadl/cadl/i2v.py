"""Illustration2Vec model and preprocessing.
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
import json
import numpy as np
from tensorflow.python.platform import gfile
import tensorflow as tf
import matplotlib.pyplot as plt
from skimage.transform import resize as imresize
from cadl.utils import download


def i2v_download():
    """Download a pretrained i2v network.

    Returns
    -------
    TYPE
        Description
    """
    model = download('https://s3.amazonaws.com/cadl/models/illust2vec.tfmodel')
    return model


def i2v_tag_download():
    """Download a pretrained i2v network.

    Returns
    -------
    TYPE
        Description
    """
    model = download('https://s3.amazonaws.com/cadl/models/illust2vec_tag.tfmodel')
    tags = download('https://s3.amazonaws.com/cadl/models/tag_list.json')
    return model, tags


def get_i2v_model():
    """Get a pretrained i2v network.

    Returns
    -------
    net : dict
        {'graph_def': graph_def, 'labels': synsets}
        where the graph_def is a tf.GraphDef and the synsets
        map an integer label from 0-1000 to a list of names
    """
    # Download the trained net
    model = i2v_download()

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

    return {'graph_def': graph_def}


def get_i2v_tag_model():
    """Get a pretrained i2v tag network.

    Returns
    -------
    net : dict
        {'graph_def': graph_def, 'labels': synsets}
        where the graph_def is a tf.GraphDef and the synsets
        map an integer label from 0-1000 to a list of names
    """
    # Download the trained net
    model, tags = i2v_tag_download()
    tags = json.load(open(tags, 'r'))

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
        'labels': tags,
        'preprocess': preprocess,
        'deprocess': deprocess
    }


def preprocess(img, crop=True, resize=True, dsize=(224, 224)):
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
    mean_img = np.array([164.76139251, 167.47864617, 181.13838569])
    if img.dtype == np.uint8:
        img = (img[..., ::-1] - mean_img).astype(np.float32)
    else:
        img = img[..., ::-1] * 255.0 - mean_img

    if crop:
        short_edge = min(img.shape[:2])
        yy = int((img.shape[0] - short_edge) / 2)
        xx = int((img.shape[1] - short_edge) / 2)
        crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    else:
        crop_img = img
    if resize:
        norm_img = imresize(crop_img, dsize, preserve_range=True)
    else:
        norm_img = crop_img
    return (norm_img).astype(np.float32)


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
    mean_img = np.array([164.76139251, 167.47864617, 181.13838569])
    processed = (img + mean_img)[..., ::-1]
    return np.clip(processed, 0, 255).astype(np.uint8)
    # return ((img / np.max(np.abs(img))) * 127.5 +
    #         127.5).astype(np.uint8)


def test_i2v():
    """Loads the i2v network and applies it to a test image.
    """
    with tf.Session() as sess:
        net = get_i2v_model()
        tf.import_graph_def(net['graph_def'], name='i2v')
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
