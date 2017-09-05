"""Basic PixelRNN i.e. CharRNN style, none of the fancy ones (i.e. Row, Diag, BiDiag).
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

Attributes
----------
B : int
    Description
C : int
    Description
ckpt_name : str
    Description
H : int
    Description
n_epochs : int
    Description
n_units : int
    Description
W : int
    Description
"""
import os
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from cadl import dataset_utils as dsu

# Parameters for training
ckpt_name = 'pixelrnn.ckpt'
n_epochs = 10
n_units = 100
B = 50
H = 32
W = 32
C = 3


def build_pixel_rnn_basic_model(B=50, H=32, W=32, C=32, n_units=100,
                                n_layers=2):
    """Summary

    Parameters
    ----------
    B : int, optional
        Description
    H : int, optional
        Description
    W : int, optional
        Description
    C : int, optional
        Description
    n_units : int, optional
        Description
    n_layers : int, optional
        Description

    Returns
    -------
    TYPE
        Description
    """
    # Input to the network, a batch of images
    X = tf.placeholder(tf.float32, shape=[B, H, W, C], name='X')
    keep_prob = tf.placeholder(tf.float32, shape=1, name='keep_prob')

    # Flatten to 2 dimensions
    X_2d = tf.reshape(X, [-1, H * W * C])

    # Turn each pixel value into a vector of one-hot values
    X_onehot = tf.one_hot(tf.cast(X_2d, tf.uint8), depth=256, axis=2)

    # Split each pixel into its own tensor resulting in H * W * C number of
    # Tensors each shaped as B x 256
    pixels = [
        tf.squeeze(p, axis=1) for p in tf.split(X_onehot, H * W * C, axis=1)
    ]

    # Create a GRU recurrent layer
    cells = tf.contrib.rnn.GRUCell(n_units)
    initial_state = cells.zero_state(
        batch_size=tf.shape(X)[0], dtype=tf.float32)
    if n_layers > 1:
        cells = tf.contrib.rnn.MultiRNNCell(
            [cells] * n_layers, state_is_tuple=True)
        initial_state = cells.zero_state(tf.shape(X)[0], tf.float32)
    cells = tf.contrib.rnn.DropoutWrapper(cells, output_keep_prob=keep_prob)

    # Connect our pixel distributions (onehots) to an rnn, this will return us a
    # list of tensors, one for each of our pixels.
    hs, final_state = tf.contrib.rnn.static_rnn(
        cells, pixels, initial_state=initial_state)

    # Concat N pixels result back into a Tensor, B x N x n_units
    stacked = tf.concat([tf.expand_dims(h_i, axis=1) for h_i in hs], axis=1)

    # And now to 2d so we can connect to FC layer
    stacked = tf.reshape(stacked, [-1, n_units])

    # And now connect to FC layer
    prediction = slim.linear(stacked, 256, scope='linear')
    if B * H * W * C > 1:
        prediction = tf.slice(prediction, [0, 0],
                              [int(prediction.shape[0] - 1), -1])
        X_onehot_flat = tf.slice(
            tf.reshape(X_onehot, [-1, 256]), [1, 0], [-1, -1])
        loss = tf.nn.softmax_cross_entropy_with_logits(
            labels=X_onehot_flat, logits=prediction)

        cost = tf.reduce_mean(loss)
    else:
        cost = None

    return {
        'X': X,
        'recon': prediction,
        'cost': cost,
        'initial_state': initial_state,
        'final_state': final_state
    }


def infer(sess, net, H, W, C, pixel_value=128, state=None):
    """Summary

    Parameters
    ----------
    sess : TYPE
        Description
    net : TYPE
        Description
    H : TYPE
        Description
    W : TYPE
        Description
    C : TYPE
        Description
    pixel_value : int, optional
        Description
    state : None, optional
        Description

    Returns
    -------
    TYPE
        Description
    """
    X = np.reshape(pixel_value, [1, 1, 1, 1])
    synthesis = [pixel_value]
    if state is None:
        state = sess.run(net['initial_state'])
    for pixel_i in range(H * W * C - 1):
        next, state = sess.run(
            [net['recon'], net['final_state']],
            feed_dict={net['X']: X,
                       net['initial_state']: state})
        synthesis.append(np.argmax(next))
    return synthesis


def train_tiny_imagenet():
    """Summary
    """
    net = build_pixel_rnn_basic_model()

    # build the optimizer (this will take a while!)
    optimizer = tf.train.AdamOptimizer(
        learning_rate=0.001).minimize(net['cost'])

    # Load a list of files for tiny imagenet, downloading if necessary
    imagenet_files = dsu.tiny_imagenet_load()

    # Create a threaded image pipeline which will load/shuffle/crop/resize
    batch = dsu.create_input_pipeline(
        imagenet_files,
        batch_size=B,
        n_epochs=n_epochs,
        shape=[64, 64, 3],
        crop_shape=[32, 32, 3],
        crop_factor=0.5,
        n_threads=8)

    sess = tf.Session()
    saver = tf.train.Saver()
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
        saver.restore(sess, tf.train.latest_checkpoint('./'))

    epoch_i = 0
    batch_i = 0
    save_step = 100
    try:
        while not coord.should_stop() and epoch_i < n_epochs:
            batch_i += 1
            batch_xs = sess.run(batch)
            train_cost = sess.run(
                [net['cost'], optimizer], feed_dict={net['X']: batch_xs})[0]
            print(batch_i, train_cost)
            if batch_i % save_step == 0:
                # Save the variables to disk.  Don't write the meta graph
                # since we can use the code to create it, and it takes a long
                # time to create the graph since it is so deep
                saver.save(
                    sess,
                    ckpt_name,
                    global_step=batch_i,
                    write_meta_graph=False)
    except tf.errors.OutOfRangeError:
        print('Done.')
    finally:
        # One of the threads has issued an exception.  So let's tell all the
        # threads to shutdown.
        coord.request_stop()

    # Wait until all threads have finished.
    coord.join(threads)

    # Clean up the session.
    sess.close()


if __name__ == '__main__':
    train_tiny_imagenet()
