"""Deep Dream using the Inception v5 network.
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
import tensorflow as tf
from scipy.ndimage.filters import gaussian_filter
from skimage.transform import resize
from scipy.misc import imsave
from cadl import inception, vgg16, i2v
from cadl import gif


def get_labels(model='inception'):
    """Return labels corresponding to the `neuron_i` parameter of deep dream.

    Parameters
    ----------
    model : str, optional
        Which model to load. Must be one of: ['inception'], 'i2v_tag', 'i2v',
        'vgg16', or 'vgg_face'.

    Raises
    ------
    ValueError
        Unknown model.  Must be one of: ['inception'], 'i2v_tag', 'i2v',
        'vgg16', or 'vgg_face'.

    Returns
    -------
    TYPE
        Description
    """
    if model == 'inception':
        net = inception.get_inception_model()
        return net['labels']
    elif model == 'i2v_tag':
        net = i2v.get_i2v_tag_model()
        return net['labels']
    elif model == 'vgg16':
        net = vgg16.get_vgg_model()
        return net['labels']
    elif model == 'vgg_face':
        net = vgg16.get_vgg_face_model()
        return net['labels']
    else:
        raise ValueError("Unknown model or this model does not have labels!")


def get_layer_names(model='inception'):
    """Retun every layer's index and name in the given model.

    Parameters
    ----------
    model : str, optional
        Which model to load. Must be one of: ['inception'], 'i2v_tag', 'i2v',
        'vgg16', or 'vgg_face'.

    Returns
    -------
    names : list of tuples
        The index and layer's name for every layer in the given model.
    """
    g = tf.Graph()
    with tf.Session(graph=g):
        if model == 'inception':
            net = inception.get_inception_model()
        elif model == 'vgg_face':
            net = vgg16.get_vgg_face_model()
        elif model == 'vgg16':
            net = vgg16.get_vgg_model()
        elif model == 'i2v':
            net = i2v.get_i2v_model()
        elif model == 'i2v-tag':
            net = i2v.get_i2v_tag_model()

        tf.import_graph_def(net['graph_def'], name='net')
        names = [(i, op.name) for i, op in enumerate(g.get_operations())]
        return names


def _setup(input_img, model, downsize):
    """Internal use only. Load the given model's graph and preprocess an image.

    Parameters
    ----------
    input_img : np.ndarray
        Image to process with the model's normalizaiton process.
    model : str
        Which model to load. Must be one of: ['inception'], 'i2v_tag', 'i2v',
        'vgg16', or 'vgg_face'.
    downsize : bool
        Optionally crop/resize the input image to the standard shape.  Only
        applies to inception network which is all convolutional.

    Returns
    -------
    net, img, preprocess, deprocess : dict, np.ndarray, function, function
        net : The networks graph_def and labels
        img : The preprocessed input image
        preprocess: Function for preprocessing an image
        deprocess: Function for deprocessing an image

    Raises
    ------
    ValueError
        If model is unknown.
    """
    if model == 'inception':
        net = inception.get_inception_model()
        img = inception.preprocess(
            input_img, resize=downsize, crop=downsize)[np.newaxis]
        deprocess, preprocess = inception.deprocess, inception.preprocess
    elif model == 'vgg_face':
        net = vgg16.get_vgg_face_model()
        img = vgg16.preprocess(input_img)[np.newaxis]
        deprocess, preprocess = vgg16.deprocess, vgg16.preprocess
    elif model == 'vgg16':
        net = vgg16.get_vgg_model()
        img = vgg16.preprocess(input_img)[np.newaxis]
        deprocess, preprocess = vgg16.deprocess, vgg16.preprocess
    elif model == 'i2v':
        net = i2v.get_i2v_model()
        img = i2v.preprocess(input_img)[np.newaxis]
        deprocess, preprocess = i2v.deprocess, i2v.preprocess
    elif model == 'i2v_tag':
        net = i2v.get_i2v_tag_model()
        img = i2v.preprocess(input_img)[np.newaxis]
        deprocess, preprocess = i2v.deprocess, i2v.preprocess
    else:
        raise ValueError("Unknown model name!  Supported: " +
                         "['inception', 'vgg_face', 'vgg16', 'i2v', 'i2v_tag']")

    return net, img, preprocess, deprocess


def _apply(img,
           gradient,
           it_i,
           decay=0.998,
           sigma=1.5,
           blur_step=10,
           step=1.0,
           crop=0,
           crop_step=1,
           pth=0):
    """Interal use only. Apply the gradient to an image with the given params.

    Parameters
    ----------
    img : np.ndarray
        Tensor to apply gradient ascent to.
    gradient : np.ndarray
        Gradient to ascend to.
    it_i : int
        Current iteration (used for step modulos)
    decay : float, optional
        Amount to decay.
    sigma : float, optional
        Sigma for Gaussian Kernel.
    blur_step : int, optional
        How often to blur.
    step : float, optional
        Step for gradient ascent.
    crop : int, optional
        Amount to crop from each border.
    crop_step : int, optional
        How often to crop.
    pth : int, optional
        Percentile to mask out.

    No Longer Returned
    ------------------
    img : np.ndarray
        Ascended image.
    """
    gradient /= (np.std(gradient) + 1e-10)
    img += gradient * step
    img *= decay

    if pth:
        mask = (np.abs(img) < np.percentile(np.abs(img), pth))
        img = img - img * mask

    if blur_step and it_i % blur_step == 0:
        for ch_i in range(3):
            img[..., ch_i] = gaussian_filter(img[..., ch_i], sigma)

    if crop and it_i % crop_step == 0:
        height, width, *ch = img[0].shape

        # Crop a 1 pixel border from height and width
        img = img[:, crop:-crop, crop:-crop, :]

        # Resize
        img = resize(
            img[0], (height, width), order=3, clip=False,
            preserve_range=True)[np.newaxis].astype(np.float32)


def deep_dream(input_img,
               downsize=False,
               model='inception',
               layer_i=-1,
               neuron_i=-1,
               n_iterations=100,
               save_gif=None,
               save_images='imgs',
               device='/cpu:0',
               **kwargs):
    """Deep Dream with the given parameters.

    Parameters
    ----------
    input_img : np.ndarray
        Image to apply deep dream to.  Should be 3-dimenionsal H x W x C
        RGB uint8 or float32.
    downsize : bool, optional
        Whether or not to downsize the image.  Only applies to
        model=='inception'.
    model : str, optional
        Which model to load.  Must be one of: ['inception'], 'i2v_tag', 'i2v',
        'vgg16', or 'vgg_face'.
    layer_i : int, optional
        Which layer to use for finding the gradient.  E.g. the softmax layer
        for inception is -1, for vgg networks it is -2.  Use the function
        "get_layer_names" to find the layer number that you need.
    neuron_i : int, optional
        Which neuron to use.  -1 for the entire layer.
    n_iterations : int, optional
        Number of iterations to dream.
    save_gif : bool, optional
        Save a GIF.
    save_images : str, optional
        Folder to save images to.
    device : str, optional
        Which device to use, e.g. ['/cpu:0'] or '/gpu:0'.
    **kwargs : dict
        See "_apply" for additional parameters.

    Returns
    -------
    imgs : list of np.array
        Images of every iteration
    """
    net, img, preprocess, deprocess = _setup(input_img, model, downsize)
    batch, height, width, *ch = img.shape

    g = tf.Graph()
    with tf.Session(graph=g) as sess, g.device(device):

        tf.import_graph_def(net['graph_def'], name='net')
        names = [op.name for op in g.get_operations()]
        input_name = names[0] + ':0'
        x = g.get_tensor_by_name(input_name)

        layer = g.get_tensor_by_name(names[layer_i] + ':0')
        layer_shape = sess.run(tf.shape(layer), feed_dict={x: img})
        layer_vec = np.ones(layer_shape) / layer_shape[-1]
        layer_vec[..., neuron_i] = 1.0 - (1.0 / layer_shape[-1])

        ascent = tf.gradients(layer, x)

        imgs = []
        for it_i in range(n_iterations):
            print(it_i, np.min(img), np.max(img))
            if neuron_i == -1:
                this_res = sess.run(ascent, feed_dict={x: img})[0]
            else:
                this_res = sess.run(
                    ascent, feed_dict={x: img,
                                       layer: layer_vec})[0]

            _apply(img, this_res, it_i, **kwargs)
            imgs.append(deprocess(img[0]))

            if save_images is not None:
                if not os.path.exists(save_images):
                    os.mkdir(save_images)
                imsave(
                    os.path.join(save_images, 'frame{}.png'.format(it_i)),
                    imgs[-1])

        if save_gif is not None:
            gif.build_gif(imgs, saveto=save_gif)

    return imgs


def guided_dream(input_img,
                 guide_img=None,
                 downsize=False,
                 layers=[162, 183, 184, 247],
                 label_i=962,
                 layer_i=-1,
                 feature_loss_weight=1.0,
                 tv_loss_weight=1.0,
                 l2_loss_weight=1.0,
                 softmax_loss_weight=1.0,
                 model='inception',
                 neuron_i=920,
                 n_iterations=100,
                 save_gif=None,
                 save_images='imgs',
                 device='/cpu:0',
                 **kwargs):
    """Deep Dream v2.  Use an optional guide image and other techniques.

    Parameters
    ----------
    input_img : np.ndarray
        Image to apply deep dream to.  Should be 3-dimenionsal H x W x C
        RGB uint8 or float32.
    guide_img : np.ndarray, optional
        Optional image to find features at different layers for.  Must pass in
        a list of layers that you want to find features for.  Then the guided
        dream will try to match this images features at those layers.
    downsize : bool, optional
        Whether or not to downsize the image.  Only applies to
        model=='inception'.
    layers : list, optional
        A list of layers to find features for in the "guide_img".
    label_i : int, optional
        Which label to use for the softmax layer.  Use the "get_labels" function
        to find the index corresponding the object of interest.  If None, not
        used.
    layer_i : int, optional
        Which layer to use for finding the gradient.  E.g. the softmax layer
        for inception is -1, for vgg networks it is -2.  Use the function
        "get_layer_names" to find the layer number that you need.
    feature_loss_weight : float, optional
        Weighting for the feature loss from the guide_img.
    tv_loss_weight : float, optional
        Total variational loss weighting.  Enforces smoothness.
    l2_loss_weight : float, optional
        L2 loss weighting.  Enforces smaller values and reduces saturation.
    softmax_loss_weight : float, optional
        Softmax loss weighting.  Must set label_i.
    model : str, optional
        Which model to load.  Must be one of: ['inception'], 'i2v_tag', 'i2v',
        'vgg16', or 'vgg_face'.
    neuron_i : int, optional
        Which neuron to use.  -1 for the entire layer.
    n_iterations : int, optional
        Number of iterations to dream.
    save_gif : bool, optional
        Save a GIF.
    save_images : str, optional
        Folder to save images to.
    device : str, optional
        Which device to use, e.g. ['/cpu:0'] or '/gpu:0'.
    **kwargs : dict
        See "_apply" for additional parameters.

    Returns
    -------
    imgs : list of np.ndarray
        Images of the dream.
    """
    net, img, preprocess, deprocess = _setup(input_img, model, downsize)
    print(img.shape, input_img.shape)
    print(img.min(), img.max())

    if guide_img is not None:
        guide_img = preprocess(guide_img.copy(), model)[np.newaxis]
        assert (guide_img.shape == img.shape)
    batch, height, width, *ch = img.shape

    g = tf.Graph()
    with tf.Session(graph=g) as sess, g.device(device):
        tf.import_graph_def(net['graph_def'], name='net')
        names = [op.name for op in g.get_operations()]
        input_name = names[0] + ':0'
        x = g.get_tensor_by_name(input_name)

        features = [names[layer_i] + ':0' for layer_i in layers]
        feature_loss = tf.Variable(0.0)
        for feature_i in features:
            layer = g.get_tensor_by_name(feature_i)
            if guide_img is None:
                feature_loss += tf.reduce_mean(layer)
            else:
                # Reshape it to 2D vector
                layer = tf.reshape(layer, [-1, 1])
                # Do the same for our guide image
                guide_layer = sess.run(layer, feed_dict={x: guide_img})
                guide_layer = guide_layer.reshape(-1, 1)
                # Now calculate their dot product
                correlation = tf.matmul(guide_layer.T, layer)
                feature_loss += feature_loss_weight * tf.reduce_mean(
                    correlation)
        softmax_loss = tf.Variable(0.0)
        if label_i is not None:
            layer = g.get_tensor_by_name(names[layer_i] + ':0')
            layer_shape = sess.run(tf.shape(layer), feed_dict={x: img})
            layer_vec = np.ones(layer_shape) / layer_shape[-1]
            layer_vec[..., neuron_i] = 1.0 - 1.0 / layer_shape[1]
            softmax_loss += softmax_loss_weight * tf.reduce_mean(
                tf.nn.l2_loss(layer - layer_vec))

        dx = tf.square(x[:, :height - 1, :width - 1, :] -
                       x[:, :height - 1, 1:, :])
        dy = tf.square(x[:, :height - 1, :width - 1, :] -
                       x[:, 1:, :width - 1, :])
        tv_loss = tv_loss_weight * tf.reduce_mean(tf.pow(dx + dy, 1.2))
        l2_loss = l2_loss_weight * tf.reduce_mean(tf.nn.l2_loss(x))

        ascent = tf.gradients(feature_loss + softmax_loss + tv_loss + l2_loss,
                              x)[0]
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        sess.run(init_op)
        imgs = []
        for it_i in range(n_iterations):
            this_res, this_feature_loss, this_softmax_loss, this_tv_loss, this_l2_loss = sess.run(
                [ascent, feature_loss, softmax_loss, tv_loss, l2_loss],
                feed_dict={x: img})
            print('feature:', this_feature_loss, 'softmax:', this_softmax_loss,
                  'tv', this_tv_loss, 'l2', this_l2_loss)

            _apply(img, -this_res, it_i, **kwargs)
            imgs.append(deprocess(img[0]))

            if save_images is not None:
                imsave(
                    os.path.join(save_images, 'frame{}.png'.format(it_i)),
                    imgs[-1])

        if save_gif is not None:
            gif.build_gif(imgs, saveto=save_gif)

    return imgs
