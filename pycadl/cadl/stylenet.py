"""Style Net w/ tests for Video Style Net.
"""
"""
Video Style Net requires OpenCV 3.0.0+ w/ Contrib for Python to be installed.

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
import numpy as np
import matplotlib.pyplot as plt
import os
from cadl import vgg16
from cadl import gif
from scipy.misc import imresize


def make_4d(img):
    """Create a 4-dimensional N x H x W x C image.

    Parameters
    ----------
    img : np.ndarray
        Given image as H x W x C or H x W.

    Returns
    -------
    img : np.ndarray
        N x H x W x C image.

    Raises
    ------
    ValueError
        Unexpected number of dimensions.
    """
    if img.ndim == 2:
        img = np.expand_dims(img[np.newaxis], 3)
    elif img.ndim == 3:
        img = img[np.newaxis]
    elif img.ndim == 4:
        return img
    else:
        raise ValueError('Incorrect dimensions for image!')
    return img


def stylize(content_img,
            style_img,
            base_img=None,
            saveto=None,
            gif_step=5,
            n_iterations=100,
            style_weight=1.0,
            content_weight=1.0):
    """Stylization w/ the given content and style images.

    Follows the approach in Leon Gatys et al.

    Parameters
    ----------
    content_img : np.ndarray
        Image to use for finding the content features.
    style_img : TYPE
        Image to use for finding the style features.
    base_img : None, optional
        Image to use for the base content.  Can be noise or an existing image.
        If None, the content image will be used.
    saveto : str, optional
        Name of GIF image to write to, e.g. "stylization.gif"
    gif_step : int, optional
        Modulo of iterations to save the current stylization.
    n_iterations : int, optional
        Number of iterations to run for.
    style_weight : float, optional
        Weighting on the style features.
    content_weight : float, optional
        Weighting on the content features.

    Returns
    -------
    stylization : np.ndarray
        Final iteration of the stylization.
    """
    # Preprocess both content and style images
    content_img = vgg16.preprocess(content_img, dsize=(224, 224))[np.newaxis]
    style_img = vgg16.preprocess(style_img, dsize=(224, 224))[np.newaxis]
    if base_img is None:
        base_img = content_img
    else:
        base_img = make_4d(vgg16.preprocess(base_img, dsize=(224, 224)))

    # Get Content and Style features
    net = vgg16.get_vgg_model()
    g = tf.Graph()
    with tf.Session(graph=g) as sess:
        tf.import_graph_def(net['graph_def'], name='vgg')
        names = [op.name for op in g.get_operations()]
        x = g.get_tensor_by_name(names[0] + ':0')
        content_layer = 'vgg/conv3_2/conv3_2:0'
        content_features = g.get_tensor_by_name(content_layer).eval(
            feed_dict={
                x: content_img,
                'vgg/dropout_1/random_uniform:0': [[1.0] * 4096],
                'vgg/dropout/random_uniform:0': [[1.0] * 4096]
            })
        style_layers = [
            'vgg/conv1_1/conv1_1:0', 'vgg/conv2_1/conv2_1:0',
            'vgg/conv3_1/conv3_1:0', 'vgg/conv4_1/conv4_1:0',
            'vgg/conv5_1/conv5_1:0'
        ]
        style_activations = []
        for style_i in style_layers:
            style_activation_i = g.get_tensor_by_name(style_i).eval(
                feed_dict={
                    x: style_img,
                    'vgg/dropout_1/random_uniform:0': [[1.0] * 4096],
                    'vgg/dropout/random_uniform:0': [[1.0] * 4096]
                })
            style_activations.append(style_activation_i)
        style_features = []
        for style_activation_i in style_activations:
            s_i = np.reshape(style_activation_i,
                             [-1, style_activation_i.shape[-1]])
            gram_matrix = np.matmul(s_i.T, s_i) / s_i.size
            style_features.append(gram_matrix.astype(np.float32))

    # Optimize both
    g = tf.Graph()
    with tf.Session(graph=g) as sess:
        net_input = tf.Variable(base_img)
        tf.import_graph_def(
            net['graph_def'], name='vgg', input_map={'images:0': net_input})

        content_loss = tf.nn.l2_loss(
            (g.get_tensor_by_name(content_layer) - content_features) /
            content_features.size)
        style_loss = np.float32(0.0)
        for style_layer_i, style_gram_i in zip(style_layers, style_features):
            layer_i = g.get_tensor_by_name(style_layer_i)
            layer_shape = layer_i.get_shape().as_list()
            layer_size = layer_shape[1] * layer_shape[2] * layer_shape[3]
            layer_flat = tf.reshape(layer_i, [-1, layer_shape[3]])
            gram_matrix = tf.matmul(tf.transpose(layer_flat),
                                    layer_flat) / layer_size
            style_loss = tf.add(style_loss,
                                tf.nn.l2_loss((gram_matrix - style_gram_i) /
                                              np.float32(style_gram_i.size)))
        loss = content_weight * content_loss + style_weight * style_loss
        optimizer = tf.train.AdamOptimizer(0.01).minimize(loss)

        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        sess.run(init_op)
        imgs = []
        for it_i in range(n_iterations):
            _, this_loss, synth = sess.run(
                [optimizer, loss, net_input],
                feed_dict={
                    'vgg/dropout_1/random_uniform:0':
                    np.ones(
                        g.get_tensor_by_name('vgg/dropout_1/random_uniform:0')
                        .get_shape().as_list()),
                    'vgg/dropout/random_uniform:0':
                    np.ones(
                        g.get_tensor_by_name('vgg/dropout/random_uniform:0')
                        .get_shape().as_list())
                })
            print(
                "iteration %d, loss: %f, range: (%f - %f)" %
                (it_i, this_loss, np.min(synth), np.max(synth)),
                end='\r')
            if it_i % gif_step == 0:
                imgs.append(np.clip(synth[0], 0, 1))
        if saveto is not None:
            gif.build_gif(imgs, saveto=saveto)
    return np.clip(synth[0], 0, 1)


def warp_img(img, dx, dy):
    """Apply the motion vectors to the given image.

    Parameters
    ----------
    img : np.ndarray
        Input image to apply motion to.
    dx : np.ndarray
        H x W matrix defining the magnitude of the X vector
    dy : np.ndarray
        H x W matrix defining the magnitude of the Y vector

    Returns
    -------
    img : np.ndarray
        Image with pixels warped according to dx, dy.
    """
    warped = img.copy()
    for row_i in range(img.shape[0]):
        for col_i in range(img.shape[1]):
            dx_i = int(np.round(dx[row_i, col_i]))
            dy_i = int(np.round(dy[row_i, col_i]))
            sample_dx = np.clip(dx_i + col_i, 0, img.shape[0] - 1)
            sample_dy = np.clip(dy_i + row_i, 0, img.shape[1] - 1)
            warped[sample_dy, sample_dx, :] = img[row_i, col_i, :]
    return warped


def test_video(style_img='arles.jpg', videodir='kurosawa'):

    has_cv2 = True
    try:
        import cv2
        has_cv2 = True
        optflow = cv2.optflow.createOptFlow_DeepFlow()
    except ImportError:
        has_cv2 = False

    style_img = plt.imread(style_img)
    content_files = [
        os.path.join(videodir, f) for f in os.listdir(videodir)
        if f.endswith('.png')
    ]
    content_img = plt.imread(content_files[0])
    style_img = imresize(style_img, (448, 448)).astype(np.float32) / 255.0
    content_img = imresize(content_img, (448, 448)).astype(np.float32) / 255.0
    if has_cv2:
        prev_lum = cv2.cvtColor(content_img, cv2.COLOR_RGB2HSV)[:, :, 2]
    else:
        prev_lum = (content_img[..., 0] * 0.3 + content_img[..., 1] * 0.59 +
                    content_img[..., 2] * 0.11)
    imgs = []
    stylized = stylize(
        content_img,
        style_img,
        content_weight=5.0,
        style_weight=0.5,
        n_iterations=50)
    plt.imsave(fname=content_files[0] + 'stylized.png', arr=stylized)
    imgs.append(stylized)
    for f in content_files[1:]:
        content_img = plt.imread(f)
        content_img = imresize(content_img, (448,
                                             448)).astype(np.float32) / 255.0
        if has_cv2:
            lum = cv2.cvtColor(content_img, cv2.COLOR_RGB2HSV)[:, :, 2]
            flow = optflow.calc(prev_lum, lum, None)
            warped = warp_img(stylized, flow[..., 0], flow[..., 1])
            stylized = stylize(
                content_img,
                style_img,
                content_weight=5.0,
                style_weight=0.5,
                base_img=warped,
                n_iterations=50)
        else:
            lum = (content_img[..., 0] * 0.3 + content_img[..., 1] * 0.59 +
                   content_img[..., 2] * 0.11)
            stylized = stylize(
                content_img,
                style_img,
                content_weight=5.0,
                style_weight=0.5,
                base_img=None,
                n_iterations=50)
        imgs.append(stylized)
        plt.imsave(fname=f + 'stylized.png', arr=stylized)
        prev_lum = lum
    return imgs


def test():
    """Test for artistic stylization.
    """
    from six.moves import urllib
    f = ('https://upload.wikimedia.org/wikipedia/commons/thumb/5/54/' +
         'Claude_Monet%2C_Impression%2C_soleil_levant.jpg/617px-Claude_Monet' +
         '%2C_Impression%2C_soleil_levant.jpg?download')
    filepath, _ = urllib.request.urlretrieve(f, f.split('/')[-1], None)
    style = plt.imread(filepath).astype(np.float32) / 255.0

    f = ('https://upload.wikimedia.org/wikipedia/commons/thumb/a/ae/' +
         'El_jard%C3%ADn_de_las_Delicias%2C_de_El_Bosco.jpg/640px-El_jard' +
         '%C3%ADn_de_las_Delicias%2C_de_El_Bosco.jpg')
    filepath, _ = urllib.request.urlretrieve(f, f.split('/')[-1], None)
    content = plt.imread(filepath).astype(np.float32) / 255.0

    stylize(content, style, n_iterations=20)


if __name__ == '__main__':
    test_video()
