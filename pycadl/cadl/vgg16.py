"""VGG16 pretrained model and VGG Face model.
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
import tensorflow as tf
import json
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize as imresize
from .utils import download


def get_vgg_face_model():
    """Summary

    Returns
    -------
    TYPE
        Description
    """
    download('https://s3.amazonaws.com/cadl/models/vgg_face.tfmodel')
    with open("vgg_face.tfmodel", mode='rb') as f:
        graph_def = tf.GraphDef()
        try:
            graph_def.ParseFromString(f.read())
        except:
            print('try adding PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python ' +
                  'to environment.  e.g.:\n' +
                  'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python ipython\n' +
                  'See here for info: ' +
                  'https://github.com/tensorflow/tensorflow/issues/582')

    download('https://s3.amazonaws.com/cadl/models/vgg_face.json')
    labels = json.load(open('vgg_face.json'))

    return {
        'graph_def': graph_def,
        'labels': labels,
        'preprocess': preprocess,
        'deprocess': deprocess
    }


def get_vgg_model():
    """Summary

    Returns
    -------
    TYPE
        Description
    """
    download('https://s3.amazonaws.com/cadl/models/vgg16.tfmodel')
    with open("vgg16.tfmodel", mode='rb') as f:
        graph_def = tf.GraphDef()
        try:
            graph_def.ParseFromString(f.read())
        except:
            print('try adding PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python ' +
                  'to environment.  e.g.:\n' +
                  'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python ipython\n' +
                  'See here for info: ' +
                  'https://github.com/tensorflow/tensorflow/issues/582')

    download('https://s3.amazonaws.com/cadl/models/synset.txt')
    with open('synset.txt') as f:
        labels = [(idx, l.strip()) for idx, l in enumerate(f.readlines())]

    return {
        'graph_def': graph_def,
        'labels': labels,
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
    if img.dtype == np.uint8:
        img = img / 255.0

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
    return np.clip(img * 255, 0, 255).astype(np.uint8)
    # return ((img / np.max(np.abs(img))) * 127.5 +
    #         127.5).astype(np.uint8)


def test_vgg():
    """Loads the VGG network and applies it to a test image.
    """
    with tf.Session() as sess:
        net = get_vgg_model()
        tf.import_graph_def(net['graph_def'], name='vgg')
        g = tf.get_default_graph()
        names = [op.name for op in g.get_operations()]
        input_name = names[0] + ':0'
        x = g.get_tensor_by_name(input_name)
        softmax = g.get_tensor_by_name(names[-2] + ':0')

        og = plt.imread('bosch.png')
        img = preprocess(og)[np.newaxis, ...]
        res = np.squeeze(softmax.eval(feed_dict={
            x: img,
            'vgg/dropout_1/random_uniform:0': [[1.0]],
            'vgg/dropout/random_uniform:0': [[1.0]]}))
        print([(res[idx], net['labels'][idx])
               for idx in res.argsort()[-5:][::-1]])

        """Let's visualize the network's gradient activation
        when backpropagated to the original input image.  This
        is effectively telling us which pixels contribute to the
        predicted class or given neuron"""
        features = [name for name in names if 'BiasAdd' in name.split()[-1]]
        from math import sqrt, ceil
        n_plots = ceil(sqrt(len(features) + 1))
        fig, axs = plt.subplots(n_plots, n_plots)
        plot_i = 0
        axs[0][0].imshow(img[0])
        for feature_i, featurename in enumerate(features):
            plot_i += 1
            feature = g.get_tensor_by_name(featurename + ':0')
            neuron = tf.reduce_max(feature, 1)
            saliency = tf.gradients(tf.reduce_sum(neuron), x)
            neuron_idx = tf.arg_max(feature, 1)
            this_res = sess.run([saliency[0], neuron_idx], feed_dict={
                x: img,
                'vgg/dropout_1/random_uniform:0': [[1.0]],
                'vgg/dropout/random_uniform:0': [[1.0]]})

            grad = this_res[0][0] / np.max(np.abs(this_res[0]))
            ax = axs[plot_i // n_plots][plot_i % n_plots]
            ax.imshow((grad * 127.5 + 127.5).astype(np.uint8))
            ax.set_title(featurename)

        """Deep Dreaming takes the backpropagated gradient activations
        and simply adds it to the image, running the same process again
        and again in a loop.  There are many tricks one can add to this
        idea, such as infinitely zooming into the image by cropping and
        scaling, adding jitter by randomly moving the image around, or
        adding constraints on the total activations."""
        og = plt.imread('street.png')
        crop = 2
        img = preprocess(og)[np.newaxis, ...]
        layer = g.get_tensor_by_name(features[3] + ':0')
        n_els = layer.get_shape().as_list()[1]
        neuron_i = np.random.randint(1000)
        layer_vec = np.zeros((1, n_els))
        layer_vec[0, neuron_i] = 1
        neuron = tf.reduce_max(layer, 1)
        saliency = tf.gradients(tf.reduce_sum(neuron), x)
        for it_i in range(3):
            print(it_i)
            this_res = sess.run(saliency[0], feed_dict={
                x: img,
                layer: layer_vec,
                'vgg/dropout_1/random_uniform:0': [[1.0]],
                'vgg/dropout/random_uniform:0': [[1.0]]})
            grad = this_res[0] / np.mean(np.abs(grad))
            img = img[:, crop:-crop - 1, crop:-crop - 1, :]
            img = imresize(img[0], (224, 224))[np.newaxis]
            img += grad
        plt.imshow(deprocess(img[0]))


def test_vgg_face():
    """Loads the VGG network and applies it to a test image.
    """
    with tf.Session() as sess:
        net = get_vgg_face_model()
        x = tf.placeholder(tf.float32, [1, 224, 224, 3], name='x')
        tf.import_graph_def(net['graph_def'], name='vgg',
                            input_map={'Placeholder:0': x})
        g = tf.get_default_graph()
        names = [op.name for op in g.get_operations()]

        og = plt.imread('bricks.png')[..., :3]
        img = preprocess(og)[np.newaxis, ...]
        plt.imshow(img[0])
        plt.show()

        """Let's visualize the network's gradient activation
        when backpropagated to the original input image.  This
        is effectively telling us which pixels contribute to the
        predicted class or given neuron"""
        features = [name for name in names if 'BiasAdd' in name.split()[-1]]
        from math import sqrt, ceil
        n_plots = ceil(sqrt(len(features) + 1))
        fig, axs = plt.subplots(n_plots, n_plots)
        plot_i = 0
        axs[0][0].imshow(img[0])
        for feature_i, featurename in enumerate(features):
            plot_i += 1
            feature = g.get_tensor_by_name(featurename + ':0')
            neuron = tf.reduce_max(feature, 1)
            saliency = tf.gradients(tf.reduce_sum(neuron), x)
            neuron_idx = tf.arg_max(feature, 1)
            this_res = sess.run([saliency[0], neuron_idx], feed_dict={x: img})

            grad = this_res[0][0] / np.max(np.abs(this_res[0]))
            ax = axs[plot_i // n_plots][plot_i % n_plots]
            ax.imshow((grad * 127.5 + 127.5).astype(np.uint8))
            ax.set_title(featurename)
            plt.waitforbuttonpress()

        """Deep Dreaming takes the backpropagated gradient activations
        and simply adds it to the image, running the same process again
        and again in a loop.  There are many tricks one can add to this
        idea, such as infinitely zooming into the image by cropping and
        scaling, adding jitter by randomly moving the image around, or
        adding constraints on the total activations."""
        og = plt.imread('street.png')
        crop = 2
        img = preprocess(og)[np.newaxis, ...]
        layer = g.get_tensor_by_name(features[3] + ':0')
        n_els = layer.get_shape().as_list()[1]
        neuron_i = np.random.randint(1000)
        layer_vec = np.zeros((1, n_els))
        layer_vec[0, neuron_i] = 1
        neuron = tf.reduce_max(layer, 1)
        saliency = tf.gradients(tf.reduce_sum(neuron), x)
        for it_i in range(3):
            print(it_i)
            this_res = sess.run(saliency[0], feed_dict={
                x: img,
                layer: layer_vec,
                'vgg/dropout_1/random_uniform:0': [[1.0]],
                'vgg/dropout/random_uniform:0': [[1.0]]})
            grad = this_res[0] / np.mean(np.abs(grad))
            img = img[:, crop:-crop - 1, crop:-crop - 1, :]
            img = imresize(img[0], (224, 224))[np.newaxis]
            img += grad
        plt.imshow(deprocess(img[0]))


if __name__ == '__main__':
    test_vgg_face()
