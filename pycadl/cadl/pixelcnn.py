"""Conditional Gated Pixel CNN.
"""
"""
Thanks to many reference implementations
----------------------------------------
https://github.com/anantzoid/Conditional-PixelCNN-decoder
https://github.com/openai/pixel-cnn
https://github.com/PrajitR/fast-pixel-cnn

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
import os
from cadl import dataset_utils as dsu


def gated_conv2d(X,
                 K_h,
                 K_w,
                 K_c,
                 strides=[1, 1, 1, 1],
                 padding='SAME',
                 mask=None,
                 cond_h=None,
                 vertical_h=None):
    """Summary

    Parameters
    ----------
    X : TYPE
        Description
    K_h : TYPE
        Description
    K_w : TYPE
        Description
    K_c : TYPE
        Description
    strides : list, optional
        Description
    padding : str, optional
        Description
    mask : None, optional
        Description
    cond_h : None, optional
        Description
    vertical_h : None, optional
        Description

    Returns
    -------
    TYPE
        Description
    """
    with tf.variable_scope('masked_cnn'):
        W = tf.get_variable(
            name='W',
            shape=[K_h, K_w, X.shape[-1].value, K_c * 2],
            initializer=tf.contrib.layers.xavier_initializer_conv2d())
        b = tf.get_variable(
            name='b', shape=[K_c * 2], initializer=tf.zeros_initializer())
        if mask is not None:
            W = tf.multiply(mask, W)
        # Initial convolution with masked kernel
        h = tf.nn.bias_add(
            tf.nn.conv2d(X, W, strides=strides, padding=padding), b)

    # Combine the horizontal stack's pre-activations to our hidden embedding before
    # applying the split nonlinearities.  Check Figure 2 for details.
    if vertical_h is not None:
        with tf.variable_scope('vtoh'):
            W_vtoh = tf.get_variable(
                name='W',
                shape=[1, 1, K_c * 2, K_c * 2],
                initializer=tf.contrib.layers.xavier_initializer_conv2d())
            b_vtob = tf.get_variable(
                name='b', shape=[K_c * 2], initializer=tf.zeros_initializer())
            h = tf.add(h,
                       tf.nn.bias_add(
                           tf.nn.conv2d(
                               vertical_h,
                               W_vtoh,
                               strides=strides,
                               padding=padding), b_vtob))

    # Condition on some given data
    if cond_h is not None:
        with tf.variable_scope('conditioning'):
            V = tf.get_variable(
                name='V',
                shape=[cond_h.shape[1].value, K_c],
                initializer=tf.contrib.layers.xavier_initializer_conv2d())
            b = tf.get_variable(
                name='b', shape=[K_c], initializer=tf.zeros_initializer())
            h = tf.add(
                h,
                tf.reshape(
                    tf.nn.bias_add(tf.matmul(cond_h, V), b),
                    tf.shape(X)[0:3] + [K_c]),
                name='h')

    with tf.variable_scope('gated_cnn'):
        # Finally slice and apply gated multiplier
        h_f = tf.slice(h, [0, 0, 0, 0], [-1, -1, -1, K_c])
        h_g = tf.slice(h, [0, 0, 0, K_c], [-1, -1, -1, K_c])
        y = tf.multiply(tf.nn.tanh(h_f), tf.sigmoid(h_g))

    return y, h


def build_conditional_pixel_cnn_model(B=None,
                                      H=32,
                                      W=32,
                                      C=3,
                                      n_conditionals=None):
    """Conditional Gated Pixel CNN Model.

    From the paper
    --------------
        van den Oord, A., Kalchbrenner, N., Vinyals, O.,
        Espeholt, L., Graves, A., & Kavukcuoglu, K. (2016).
        Conditional Image Generation with PixelCNN Decoders.

    Implements most of the paper, except for the autoencoder,
    triplet loss of face embeddings, and pad/crop/shift ops for
    convolution (as it is not as clear IMO from a pedagogical
    point of view).

    Parameters
    ----------
    B : None, optional
        Description
    H : int, optional
        Description
    W : int, optional
        Description
    C : int, optional
        Description
    n_conditionals : None, optional
        Description

    Returns
    -------
    TYPE
        Description
    """
    n_conditionals = None
    X = tf.placeholder(name='X', dtype=tf.uint8, shape=[None, H, W, C])

    X_ = (tf.cast(X, tf.float32) - 127.5) / 2.0

    n_layers = 10
    D = 256
    fmaps = 64

    K_hs = [7] + [3] * (n_layers - 1)
    K_ws = [7 * C] + [3 * C] * (n_layers - 1)
    K_cs = [fmaps] * n_layers

    if n_conditionals is not None:
        cond_h = tf.placeholder(
            name='cond_h', dtype=tf.float32, shape=[None, n_conditionals])
    else:
        cond_h = None

    vertical_X = X_
    horizontal_X = X_
    for K_h, K_w, K_c, layer_i in zip(K_hs, K_ws, K_cs, range(n_layers)):

        # Create two masks: one for the first layer (a) and another for all
        # other layers (b). Really dumb names but am just following the paper.
        # See Figure 2 of Pixel Recurrent Neural Networks for more info.
        if layer_i == 0:
            mask = np.ones((K_h, K_w, 1, 1), dtype=np.float32)
            mask[(K_h // 2 + 1):, :, :, :] = 0.0
            mask[K_h // 2, K_w // 2:, :, :] = 0.0
        else:
            mask = np.ones((K_h, K_w, 1, 1), dtype=np.float32)
            mask[(K_h // 2 + 1):, :, :, :] = 0.0
            mask[K_h // 2, (K_w // 2 + 1):, :, :] = 0.0

        with tf.variable_scope('layer_{}'.format(layer_i)):
            # Vertical layer
            with tf.variable_scope('vertical'):
                vertical_Y, vertical_h = gated_conv2d(
                    vertical_X, K_h, K_w, K_c, mask=mask, cond_h=cond_h)

            # Horizontal layer
            with tf.variable_scope('horizontal'):

                # Gated convolution adding in vertical stack information
                horizontal_Y, horizontal_h = gated_conv2d(
                    horizontal_X,
                    1,
                    K_w,
                    K_c,
                    mask=mask[K_h // 2, :, :, :],
                    vertical_h=vertical_h,
                    cond_h=cond_h)

                # 1x1 to reduce channels
                with tf.variable_scope('1x1'):
                    W_1x1 = tf.get_variable(
                        name='W',
                        shape=[1, 1, K_c, D],
                        initializer=tf.contrib.layers.xavier_initializer_conv2d())
                    b_1x1 = tf.get_variable(
                        name='b', shape=[D], initializer=tf.ones_initializer())
                    horizontal_Y = tf.nn.bias_add(
                        tf.nn.conv2d(
                            horizontal_Y,
                            W_1x1,
                            strides=[1, 1, 1, 1],
                            padding='SAME'), b_1x1)

                # Add Residual
                if layer_i > 0:
                    with tf.variable_scope('residual'):
                        horizontal_Y = tf.add(horizontal_X, horizontal_Y)

            vertical_X = vertical_Y
            horizontal_X = horizontal_Y

    # ReLu followed by 1x1 conv for 2 layers:
    # 1x1 to reduce channels
    Y = horizontal_X
    with tf.variable_scope('output/1x1_1'):
        W_1x1 = tf.get_variable(
            name='W',
            shape=[1, 1, D, D],
            initializer=tf.contrib.layers.xavier_initializer_conv2d())
        b_1x1 = tf.get_variable(
            name='b', shape=[D], initializer=tf.ones_initializer())
        Y = tf.nn.relu(
            tf.nn.bias_add(
                tf.nn.conv2d(Y, W_1x1, strides=[1, 1, 1, 1], padding='SAME'),
                b_1x1))

    with tf.variable_scope('output/1x1_2'):
        W_1x1 = tf.get_variable(
            name='W',
            shape=[1, 1, D, D * C],
            initializer=tf.contrib.layers.xavier_initializer_conv2d())
        b_1x1 = tf.get_variable(
            name='b', shape=[D * C], initializer=tf.ones_initializer())
        Y = tf.nn.bias_add(
            tf.nn.conv2d(Y, W_1x1, strides=[1, 1, 1, 1], padding='SAME'), b_1x1)
        Y = tf.reshape(Y, [-1, D])

    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=Y, labels=tf.cast(tf.reshape(X, [-1]), tf.int32))
    cost = tf.reduce_mean(loss)
    preds = tf.nn.softmax(Y)
    sampled_preds = tf.multinomial(Y, num_samples=1)

    tf.summary.image('actual', X)
    tf.summary.image('preds',
                     tf.reshape(
                         tf.cast(tf.argmax(Y, axis=1), tf.uint8), (-1, H, W,
                                                                   C)))
    tf.summary.histogram('loss', loss)
    tf.summary.scalar('cost', cost)
    summaries = tf.summary.merge_all()

    return {
        'cost': cost,
        'X': X,
        'preds': preds,
        'sampled_preds': sampled_preds,
        'summaries': summaries
    }


def train_tiny_imagenet(ckpt_path='pixelcnn',
                        n_epochs=1000,
                        save_step=100,
                        write_step=25,
                        B=32,
                        H=64,
                        W=64,
                        C=3):
    """Summary

    Parameters
    ----------
    ckpt_path : str, optional
        Description
    n_epochs : int, optional
        Description
    save_step : int, optional
        Description
    write_step : int, optional
        Description
    B : int, optional
        Description
    H : int, optional
        Description
    W : int, optional
        Description
    C : int, optional
        Description
    """
    ckpt_name = os.path.join(ckpt_path, 'pixelcnn.ckpt')

    with tf.Graph().as_default(), tf.Session() as sess:
        # Not actually conditioning on anything here just using the gated cnn model
        net = build_conditional_pixel_cnn_model(B=B, H=H, W=W, C=C)

        # build the optimizer (this will take a while!)
        optimizer = tf.train.AdamOptimizer(
            learning_rate=0.001).minimize(net['cost'])

        # Load a list of files for tiny imagenet, downloading if necessary
        imagenet_files = dsu.tiny_imagenet_load()

        # Create a threaded image pipeline which will load/shuffle/crop/resize
        batch = dsu.create_input_pipeline(
            imagenet_files[0],
            batch_size=B,
            n_epochs=n_epochs,
            shape=[64, 64, 3],
            crop_shape=[H, W, C],
            crop_factor=1.0,
            n_threads=8)

        saver = tf.train.Saver()
        writer = tf.summary.FileWriter(ckpt_path)
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        sess.run(init_op)

        # This will handle our threaded image pipeline
        coord = tf.train.Coordinator()

        # Ensure no more changes to graph
        tf.get_default_graph().finalize()

        # Start up the queues for handling the image pipeline
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        if os.path.exists(ckpt_name + '.index') or os.path.exists(ckpt_name):
            saver.restore(sess, ckpt_name)
            saver.restore(sess, tf.train.latest_checkpoint(ckpt_path))

        epoch_i = 0
        batch_i = 0
        try:
            while not coord.should_stop() and epoch_i < n_epochs:
                batch_i += 1
                batch_xs = sess.run(batch)
                train_cost = sess.run(
                    [net['cost'], optimizer], feed_dict={net['X']: batch_xs})[0]

                print(batch_i, train_cost)
                if batch_i % write_step == 0:
                    summary = sess.run(
                        net['summaries'], feed_dict={net['X']: batch_xs})
                    writer.add_summary(summary, batch_i)

                if batch_i % save_step == 0:
                    # Save the variables to disk.  Don't write the meta graph
                    # since we can use the code to create it, and it takes a long
                    # time to create the graph since it is so deep
                    saver.save(
                        sess,
                        ckpt_name,
                        global_step=batch_i,
                        write_meta_graph=True)
        except tf.errors.OutOfRangeError:
            print('Done.')
        finally:
            # One of the threads has issued an exception.  So let's tell all the
            # threads to shutdown.
            coord.request_stop()

        # Wait until all threads have finished.
        coord.join(threads)


def generate():
    """Summary
    """
    # Parameters for generation
    ckpt_path = 'pixelcnn'
    B = None
    H = 64
    W = 64
    C = 3

    with tf.Graph().as_default(), tf.Session() as sess:
        # Not actually conditioning on anything here just using the gated cnn model
        net = build_conditional_pixel_cnn_model(B=B, H=H, W=W, C=C)

        # Load a list of files for tiny imagenet, downloading if necessary
        imagenet_files = dsu.tiny_imagenet_load()

        saver = tf.train.Saver()
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        sess.run(init_op)
        saver.restore(sess, tf.train.latest_checkpoint(ckpt_path))

        import matplotlib.pyplot as plt
        img = plt.imread(imagenet_files[0][1000])
        from scipy.misc import imresize
        og_img = imresize(img, (H, W))
        img = og_img.copy()
        # Zero out bottom half of image and let's try to synthesize it
        img[H // 2:, :, :] = 0

        for h_i in range(H // 2, H):
            for w_i in range(W):
                for c_i in range(C):
                    print(h_i, w_i, c_i, end='\r')
                    X = img.copy()
                    preds = sess.run(
                        net['sampled_preds'],
                        feed_dict={net['X']: X[np.newaxis]})
                    X = preds.reshape((1, H, W, C)).astype(np.uint8)
                    img[h_i, w_i, c_i] = X[0, h_i, w_i, c_i]

        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(og_img)
        axs[1].imshow(img)


if __name__ == '__main__':
    train_tiny_imagenet()
