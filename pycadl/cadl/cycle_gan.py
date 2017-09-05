"""Cycle Generative Adversarial Network for Unpaired Image to Image translation.
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
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.layers as tfl
from cadl.utils import imcrop_tosquare
from scipy.misc import imresize


def l1loss(x, y):
    """Summary

    Parameters
    ----------
    x : TYPE
        Description
    y : TYPE
        Description

    Returns
    -------
    TYPE
        Description
    """
    return tf.reduce_mean(tf.abs(x - y))


def l2loss(x, y):
    """Summary

    Parameters
    ----------
    x : TYPE
        Description
    y : TYPE
        Description

    Returns
    -------
    TYPE
        Description
    """
    return tf.reduce_mean(tf.squared_difference(x, y))


def lrelu(x, leak=0.2, name="lrelu"):
    """Summary

    Parameters
    ----------
    x : TYPE
        Description
    leak : float, optional
        Description
    name : str, optional
        Description

    Returns
    -------
    TYPE
        Description
    """
    with tf.variable_scope(name):
        return tf.maximum(x, leak * x)


def instance_norm(x, epsilon=1e-5):
    """Instance Normalization.

    See Ulyanov, D., Vedaldi, A., & Lempitsky, V. (2016).
    Instance Normalization: The Missing Ingredient for Fast Stylization,
    Retrieved from http://arxiv.org/abs/1607.08022

    Parameters
    ----------
    x : TYPE
        Description
    epsilon : float, optional
        Description

    Returns
    -------
    TYPE
        Description
    """
    with tf.variable_scope('instance_norm'):
        mean, var = tf.nn.moments(x, [1, 2], keep_dims=True)
        scale = tf.get_variable(
            name='scale',
            shape=[x.get_shape()[-1]],
            initializer=tf.truncated_normal_initializer(mean=1.0, stddev=0.02))
        offset = tf.get_variable(
            name='offset',
            shape=[x.get_shape()[-1]],
            initializer=tf.constant_initializer(0.0))
        out = scale * tf.div(x - mean, tf.sqrt(var + epsilon)) + offset
        return out


def conv2d(inputs,
           activation_fn=lrelu,
           normalizer_fn=instance_norm,
           scope='conv2d',
           **kwargs):
    """Summary

    Parameters
    ----------
    inputs : TYPE
        Description
    activation_fn : TYPE, optional
        Description
    normalizer_fn : TYPE, optional
        Description
    scope : str, optional
        Description
    **kwargs
        Description

    Returns
    -------
    TYPE
        Description
    """
    with tf.variable_scope(scope or 'conv2d'):
        h = tfl.conv2d(
            inputs=inputs,
            activation_fn=None,
            normalizer_fn=None,
            weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
            biases_initializer=None,
            **kwargs)
        if normalizer_fn:
            h = normalizer_fn(h)
        if activation_fn:
            h = activation_fn(h)
        return h


def conv2d_transpose(inputs,
                     activation_fn=lrelu,
                     normalizer_fn=instance_norm,
                     scope='conv2d_transpose',
                     **kwargs):
    """Summary

    Parameters
    ----------
    inputs : TYPE
        Description
    activation_fn : TYPE, optional
        Description
    normalizer_fn : TYPE, optional
        Description
    scope : str, optional
        Description
    **kwargs
        Description

    Returns
    -------
    TYPE
        Description
    """
    with tf.variable_scope(scope or 'conv2d_transpose'):
        h = tfl.conv2d_transpose(
            inputs=inputs,
            activation_fn=None,
            normalizer_fn=None,
            weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
            biases_initializer=None,
            **kwargs)
        if normalizer_fn:
            h = normalizer_fn(h)
        if activation_fn:
            h = activation_fn(h)
        return h


def residual_block(x, n_channels=128, kernel_size=3, scope=None):
    """Summary

    Parameters
    ----------
    x : TYPE
        Description
    n_channels : int, optional
        Description
    kernel_size : int, optional
        Description
    scope : None, optional
        Description

    Returns
    -------
    TYPE
        Description
    """
    with tf.variable_scope(scope or 'residual'):
        h = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
        h = conv2d(
            inputs=h,
            num_outputs=n_channels,
            kernel_size=kernel_size,
            padding='VALID',
            scope='1')
        h = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
        h = conv2d(
            inputs=h,
            num_outputs=n_channels,
            kernel_size=kernel_size,
            activation_fn=None,
            padding='VALID',
            scope='2')
        h = tf.add(x, h)
    return h


def encoder(x, n_filters=32, k_size=3, scope=None):
    """Summary

    Parameters
    ----------
    x : TYPE
        Description
    n_filters : int, optional
        Description
    k_size : int, optional
        Description
    scope : None, optional
        Description

    Returns
    -------
    TYPE
        Description
    """
    with tf.variable_scope(scope or 'encoder'):
        h = tf.pad(x, [[0, 0], [k_size, k_size], [k_size, k_size], [0, 0]],
                   "REFLECT")
        h = conv2d(
            inputs=h,
            num_outputs=n_filters,
            kernel_size=7,
            activation_fn=tf.nn.relu,
            stride=1,
            padding='VALID',
            scope='1')
        h = conv2d(
            inputs=h,
            num_outputs=n_filters * 2,
            kernel_size=k_size,
            activation_fn=tf.nn.relu,
            stride=2,
            scope='2')
        h = conv2d(
            inputs=h,
            num_outputs=n_filters * 4,
            kernel_size=k_size,
            activation_fn=tf.nn.relu,
            stride=2,
            scope='3')
    return h


def transform(x, img_size=256):
    """Summary

    Parameters
    ----------
    x : TYPE
        Description
    img_size : int, optional
        Description

    Returns
    -------
    TYPE
        Description
    """
    h = x
    if img_size >= 256:
        n_blocks = 9
    else:
        n_blocks = 6
    for block_i in range(n_blocks):
        with tf.variable_scope('block_{}'.format(block_i)):
            h = residual_block(h)
    return h


def decoder(x, n_filters=32, k_size=3, scope=None):
    """Summary

    Parameters
    ----------
    x : TYPE
        Description
    n_filters : int, optional
        Description
    k_size : int, optional
        Description
    scope : None, optional
        Description

    Returns
    -------
    TYPE
        Description
    """
    with tf.variable_scope(scope or 'decoder'):
        h = conv2d_transpose(
            inputs=x,
            num_outputs=n_filters * 2,
            kernel_size=k_size,
            activation_fn=tf.nn.relu,
            stride=2,
            scope='1')
        h = conv2d_transpose(
            inputs=h,
            num_outputs=n_filters,
            kernel_size=k_size,
            activation_fn=tf.nn.relu,
            stride=2,
            scope='2')
        h = tf.pad(h, [[0, 0], [k_size, k_size], [k_size, k_size], [0, 0]],
                   "REFLECT")
        h = conv2d(
            inputs=h,
            num_outputs=3,
            kernel_size=7,
            stride=1,
            padding='VALID',
            activation_fn=tf.nn.tanh,
            scope='3')
    return h


def generator(x, scope=None, reuse=None):
    """Summary

    Parameters
    ----------
    x : TYPE
        Description
    scope : None, optional
        Description
    reuse : None, optional
        Description

    Returns
    -------
    TYPE
        Description
    """
    img_size = x.get_shape().as_list()[1]
    with tf.variable_scope(scope or 'generator', reuse=reuse):
        h = encoder(x)
        h = transform(h, img_size)
        h = decoder(h)
    return h


def discriminator(x, n_filters=64, k_size=4, scope=None, reuse=None):
    """Summary

    Parameters
    ----------
    x : TYPE
        Description
    n_filters : int, optional
        Description
    k_size : int, optional
        Description
    scope : None, optional
        Description
    reuse : None, optional
        Description

    Returns
    -------
    TYPE
        Description
    """
    with tf.variable_scope(scope or 'discriminator', reuse=reuse):
        h = conv2d(
            inputs=x,
            num_outputs=n_filters,
            kernel_size=k_size,
            stride=2,
            normalizer_fn=None,
            scope='1')
        h = conv2d(
            inputs=h,
            num_outputs=n_filters * 2,
            kernel_size=k_size,
            stride=2,
            scope='2')
        h = conv2d(
            inputs=h,
            num_outputs=n_filters * 4,
            kernel_size=k_size,
            stride=2,
            scope='3')
        h = conv2d(
            inputs=h,
            num_outputs=n_filters * 8,
            kernel_size=k_size,
            stride=1,
            scope='4')
        h = conv2d(
            inputs=h,
            num_outputs=1,
            kernel_size=k_size,
            stride=1,
            activation_fn=tf.nn.sigmoid,
            scope='5')
        return h


def cycle_gan(img_size=256):
    """Summary

    Parameters
    ----------
    img_size : int, optional
        Description

    Returns
    -------
    TYPE
        Description
    """
    X_real = tf.placeholder(
        name='X', shape=[1, img_size, img_size, 3], dtype=tf.float32)
    Y_real = tf.placeholder(
        name='Y', shape=[1, img_size, img_size, 3], dtype=tf.float32)
    X_fake_sample = tf.placeholder(
        name='X_fake_sample',
        shape=[None, img_size, img_size, 3],
        dtype=tf.float32)
    Y_fake_sample = tf.placeholder(
        name='Y_fake_sample',
        shape=[None, img_size, img_size, 3],
        dtype=tf.float32)

    X_fake = generator(Y_real, scope='G_yx')
    Y_fake = generator(X_real, scope='G_xy')
    X_cycle = generator(Y_fake, scope='G_yx', reuse=True)
    Y_cycle = generator(X_fake, scope='G_xy', reuse=True)

    D_X_real = discriminator(X_real, scope='D_X')
    D_Y_real = discriminator(Y_real, scope='D_Y')
    D_X_fake = discriminator(X_fake, scope='D_X', reuse=True)
    D_Y_fake = discriminator(Y_fake, scope='D_Y', reuse=True)
    D_X_fake_sample = discriminator(X_fake_sample, scope='D_X', reuse=True)
    D_Y_fake_sample = discriminator(Y_fake_sample, scope='D_Y', reuse=True)

    # Create losses for generators
    l1 = 10.0
    loss_cycle_X = l1 * l1loss(X_real, X_cycle)
    loss_cycle_Y = l1 * l1loss(Y_real, Y_cycle)
    loss_G_xy = l2loss(D_Y_fake, 1.0)
    loss_G_yx = l2loss(D_X_fake, 1.0)
    loss_G = loss_G_xy + loss_G_yx + loss_cycle_X + loss_cycle_Y

    # Create losses for discriminators
    loss_D_Y = l2loss(D_Y_real, 1.0) + l2loss(D_Y_fake_sample, 0.0)
    loss_D_X = l2loss(D_X_real, 1.0) + l2loss(D_X_fake_sample, 0.0)

    # Summaries for monitoring training
    tf.summary.histogram("D_X_real", D_X_real)
    tf.summary.histogram("D_Y_real", D_Y_real)
    tf.summary.histogram("D_X_fake", D_X_fake)
    tf.summary.histogram("D_Y_fake", D_Y_fake)
    tf.summary.histogram("D_X_fake_sample", D_X_fake_sample)
    tf.summary.histogram("D_Y_fake_sample", D_Y_fake_sample)
    tf.summary.image("X_real", X_real, max_outputs=1)
    tf.summary.image("Y_real", Y_real, max_outputs=1)
    tf.summary.image("X_fake", X_fake, max_outputs=1)
    tf.summary.image("Y_fake", Y_fake, max_outputs=1)
    tf.summary.image("X_cycle", X_cycle, max_outputs=1)
    tf.summary.image("Y_cycle", Y_cycle, max_outputs=1)
    tf.summary.histogram("X_real", X_real)
    tf.summary.histogram("Y_real", Y_real)
    tf.summary.histogram("X_fake", X_fake)
    tf.summary.histogram("Y_fake", Y_fake)
    tf.summary.histogram("X_cycle", X_cycle)
    tf.summary.histogram("Y_cycle", Y_cycle)
    tf.summary.scalar("loss_D_X", loss_D_X)
    tf.summary.scalar("loss_D_Y", loss_D_Y)
    tf.summary.scalar("loss_cycle_X", loss_cycle_X)
    tf.summary.scalar("loss_cycle_Y", loss_cycle_Y)
    tf.summary.scalar("loss_G", loss_G)
    tf.summary.scalar("loss_G_xy", loss_G_xy)
    tf.summary.scalar("loss_G_yx", loss_G_yx)
    summaries = tf.summary.merge_all()

    training_vars = tf.trainable_variables()
    D_X_vars = [v for v in training_vars if v.name.startswith('D_X')]
    D_Y_vars = [v for v in training_vars if v.name.startswith('D_Y')]
    G_xy_vars = [v for v in training_vars if v.name.startswith('G_xy')]
    G_yx_vars = [v for v in training_vars if v.name.startswith('G_yx')]
    G_vars = G_xy_vars + G_yx_vars

    return locals()


def get_images(path1, path2, img_size=256):
    """Summary

    Parameters
    ----------
    path1 : TYPE
        Description
    path2 : TYPE
        Description
    img_size : int, optional
        Description

    Returns
    -------
    TYPE
        Description
    """
    files1 = [os.path.join(path1, f) for f in os.listdir(path1)]
    files2 = [os.path.join(path2, f) for f in os.listdir(path2)]
    imgs1 = []
    for f in files1:
        try:
            img = imresize(imcrop_tosquare(plt.imread(f)), (img_size, img_size))
        except:
            continue
        if img.ndim == 3:
            imgs1.append(img[..., :3])
        else:
            img = img[..., np.newaxis]
            imgs1.append(np.concatenate([img] * 3, 2))
    imgs1 = np.array(imgs1) / 127.5 - 1.0
    imgs2 = []
    for f in files2:
        try:
            img = imresize(imcrop_tosquare(plt.imread(f)), (img_size, img_size))
        except:
            continue
        if img.ndim == 3:
            imgs2.append(img[..., :3])
        else:
            img = img[..., np.newaxis]
            imgs2.append(np.concatenate([img] * 3, 2))
    imgs2 = np.array(imgs2) / 127.5 - 1.0

    return imgs1, imgs2


def batch_generator_dataset(imgs1, imgs2):
    """Summary

    Parameters
    ----------
    imgs1 : TYPE
        Description
    imgs2 : TYPE
        Description

    Yields
    ------
    TYPE
        Description
    """
    n_imgs = min(len(imgs1), len(imgs2))
    rand_idxs1 = np.random.permutation(np.arange(len(imgs1)))[:n_imgs]
    rand_idxs2 = np.random.permutation(np.arange(len(imgs2)))[:n_imgs]
    for idx1, idx2 in zip(rand_idxs1, rand_idxs2):
        yield imgs1[[idx1]], imgs2[[idx2]]


def batch_generator_random_crop(X, Y, min_size=256, max_size=512, n_images=100):
    """Summary

    Parameters
    ----------
    X : TYPE
        Description
    Y : TYPE
        Description
    min_size : int, optional
        Description
    max_size : int, optional
        Description
    n_images : int, optional
        Description

    Yields
    ------
    TYPE
        Description
    """
    r, c, d = X.shape
    Xs, Ys = [], []
    for img_i in range(n_images):
        size = np.random.randint(min_size, max_size)
        max_r = r - size
        max_c = c - size
        this_r = np.random.randint(0, max_r)
        this_c = np.random.randint(0, max_c)
        img = imresize(X[this_r:this_r + size, this_c:this_c + size, :],
                       (min_size, min_size))
        Xs.append(img)
        img = imresize(Y[this_r:this_r + size, this_c:this_c + size, :],
                       (min_size, min_size))
        Ys.append(img)
    imgs1, imgs2 = np.array(Xs) / 127.5 - 1.0, np.array(Ys) / 127.5 - 1.0
    n_imgs = min(len(imgs1), len(imgs2))
    rand_idxs1 = np.random.permutation(np.arange(len(imgs1)))[:n_imgs]
    rand_idxs2 = np.random.permutation(np.arange(len(imgs2)))[:n_imgs]
    for idx1, idx2 in zip(rand_idxs1, rand_idxs2):
        yield imgs1[[idx1]], imgs2[[idx2]]


def train(ds_X,
          ds_Y,
          ckpt_path='cycle_gan',
          learning_rate=0.0002,
          n_epochs=100,
          img_size=256):
    """Summary

    Parameters
    ----------
    ds_X : TYPE
        Description
    ds_Y : TYPE
        Description
    ckpt_path : str, optional
        Description
    learning_rate : float, optional
        Description
    n_epochs : int, optional
        Description
    img_size : int, optional
        Description
    """
    if ds_X.ndim == 3:
        batch_generator = batch_generator_random_crop
    else:
        batch_generator = batch_generator_dataset

    # How many fake generations to keep around
    capacity = 50

    # Storage for fake generations
    fake_Xs = capacity * [
        np.zeros((1, img_size, img_size, 3), dtype=np.float32)
    ]
    fake_Ys = capacity * [
        np.zeros((1, img_size, img_size, 3), dtype=np.float32)
    ]
    idx = 0
    it_i = 0

    # Train
    with tf.Graph().as_default(), tf.Session() as sess:
        # Load the network
        net = cycle_gan(img_size=img_size)

        # Build optimizers
        D_X = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(
            net['loss_D_X'], var_list=net['D_X_vars'])
        D_Y = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(
            net['loss_D_Y'], var_list=net['D_Y_vars'])
        G = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(
            net['loss_G'], var_list=net['G_vars'])

        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        sess.run(init_op)
        saver = tf.train.Saver()
        writer = tf.summary.FileWriter(ckpt_path)
        for epoch_i in range(n_epochs):
            for X, Y in batch_generator(ds_X, ds_Y):

                # First generate in both directions
                X_fake, Y_fake = sess.run(
                    [net['X_fake'], net['Y_fake']],
                    feed_dict={net['X_real']: X,
                               net['Y_real']: Y})

                # Now sample from history
                if it_i < capacity:
                    # Not enough samples yet, fill up history buffer
                    fake_Xs[idx] = X_fake
                    fake_Ys[idx] = Y_fake
                    idx = (idx + 1) % capacity
                elif np.random.random() > 0.5:
                    # Swap out a random idx from history
                    rand_idx = np.random.randint(0, capacity)
                    fake_Xs[rand_idx], X_fake = X_fake, fake_Xs[rand_idx]
                    fake_Ys[rand_idx], Y_fake = Y_fake, fake_Ys[rand_idx]
                else:
                    # Use current generation
                    pass

                # Optimize G Networks
                loss_G = sess.run(
                    [net['loss_G'], G],
                    feed_dict={
                        net['X_real']: X,
                        net['Y_real']: Y,
                        net['Y_fake_sample']: Y_fake,
                        net['X_fake_sample']: X_fake
                    })[0]

                # Optimize D_Y
                loss_D_Y = sess.run(
                    [net['loss_D_Y'], D_Y],
                    feed_dict={
                        net['X_real']: X,
                        net['Y_real']: Y,
                        net['Y_fake_sample']: Y_fake
                    })[0]

                # Optimize D_X
                loss_D_X = sess.run(
                    [net['loss_D_X'], D_X],
                    feed_dict={
                        net['X_real']: X,
                        net['Y_real']: Y,
                        net['X_fake_sample']: X_fake
                    })[0]

                print(it_i, 'G:', loss_G, 'D_X:', loss_D_X, 'D_Y:', loss_D_Y)

                # Update summaries
                if it_i % 100 == 0:
                    summary = sess.run(
                        net['summaries'],
                        feed_dict={
                            net['X_real']: X,
                            net['Y_real']: Y,
                            net['X_fake_sample']: X_fake,
                            net['Y_fake_sample']: Y_fake
                        })
                    writer.add_summary(summary, it_i)
                it_i += 1

            # Save
            if epoch_i % 50 == 0:
                saver.save(
                    sess,
                    os.path.join(ckpt_path, 'model.ckpt'),
                    global_step=epoch_i)
