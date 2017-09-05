"""Various utilities for training WaveNet.
"""
"""
WaveNet Training code and utilities are licensed under APL from the

Google Magenta project
----------------------
https://github.com/tensorflow/magenta/blob/master/magenta/models/nsynth/wavenet

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


def shift_right(X):
    """Shift the input over by one and a zero to the front.

    Parameters
    ----------
    X
        The [mb, time, channels] tensor input.

    Returns
    -------
    x_sliced
        The [mb, time, channels] tensor output.
    """
    shape = X.get_shape().as_list()
    x_padded = tf.pad(X, [[0, 0], [1, 0], [0, 0]])
    x_sliced = tf.slice(x_padded, [0, 0, 0], tf.stack([-1, shape[1], -1]))
    x_sliced.set_shape(shape)
    return x_sliced


def mul_or_none(a, b):
    """Return the element wise multiplicative of the inputs.
    If either input is None, we return None.

    Parameters
    ----------
    a
        A tensor input.
    b
        Another tensor input with the same type as a.

    Returns
    -------
    None if either input is None. Otherwise returns a * b.
    """
    if a is None or b is None:
        return None
    return a * b


def time_to_batch(X, block_size):
    """Splits time dimension (i.e. dimension 1) of `X` into batches.
    Within each batch element, the `k*block_size` time steps are transposed,
    so that the `k` time steps in each output batch element are offset by
    `block_size` from each other.
    The number of input time steps must be a multiple of `block_size`.

    Parameters
    ----------
    X
        Tensor of shape [nb, k*block_size, n] for some natural number k.
    block_size
        number of time steps (i.e. size of dimension 1) in the output
        tensor.

    Returns
    -------
    Tensor of shape [nb*block_size, k, n]
    """
    shape = X.get_shape().as_list()
    y = tf.reshape(X, [
        shape[0], shape[1] // block_size, block_size, shape[2]
    ])
    y = tf.transpose(y, [0, 2, 1, 3])
    y = tf.reshape(y, [
        shape[0] * block_size, shape[1] // block_size, shape[2]
    ])
    y.set_shape([
        mul_or_none(shape[0], block_size), mul_or_none(shape[1], 1. / block_size),
        shape[2]
    ])
    return y


def batch_to_time(X, block_size):
    """Inverse of `time_to_batch(X, block_size)`.

    Parameters
    ----------
    X
        Tensor of shape [nb*block_size, k, n] for some natural number k.
    block_size
        number of time steps (i.e. size of dimension 1) in the output
        tensor.

    Returns
    -------
    Tensor of shape [nb, k*block_size, n].
    """
    shape = X.get_shape().as_list()
    y = tf.reshape(X, [shape[0] // block_size, block_size, shape[1], shape[2]])
    y = tf.transpose(y, [0, 2, 1, 3])
    y = tf.reshape(y, [shape[0] // block_size, shape[1] * block_size, shape[2]])
    y.set_shape([mul_or_none(shape[0], 1. / block_size),
                 mul_or_none(shape[1], block_size),
                 shape[2]])
    return y


def conv1d(X,
           num_filters,
           filter_length,
           name,
           dilation=1,
           causal=True,
           kernel_initializer=tf.uniform_unit_scaling_initializer(1.0),
           biases_initializer=tf.constant_initializer(0.0)):
    """Fast 1D convolution that supports causal padding and dilation.

    Parameters
    ----------
    X
        The [mb, time, channels] float tensor that we convolve.
    num_filters
        The number of filter maps in the convolution.
    filter_length
        The integer length of the filter.
    name
        The name of the scope for the variables.
    dilation
        The amount of dilation.
    causal
        Whether or not this is a causal convolution.
    kernel_initializer
        The kernel initialization function.
    biases_initializer
        The biases initialization function.

    Returns
    -------
    y
        The output of the 1D convolution.
    """
    batch_size, length, num_input_channels = X.get_shape().as_list()
    assert length % dilation == 0

    kernel_shape = [1, filter_length, num_input_channels, num_filters]
    strides = [1, 1, 1, 1]
    biases_shape = [num_filters]
    padding = 'VALID' if causal else 'SAME'

    with tf.variable_scope(name):
        weights = tf.get_variable(
            'W', shape=kernel_shape, initializer=kernel_initializer)
        biases = tf.get_variable(
            'biases', shape=biases_shape, initializer=biases_initializer)

    x_ttb = time_to_batch(X, dilation)
    if filter_length > 1 and causal:
        x_ttb = tf.pad(x_ttb, [[0, 0], [filter_length - 1, 0], [0, 0]])

    x_ttb_shape = x_ttb.get_shape().as_list()
    x_4d = tf.reshape(x_ttb, [x_ttb_shape[0], 1,
                              x_ttb_shape[1], num_input_channels])
    y = tf.nn.conv2d(x_4d, weights, strides, padding=padding)
    y = tf.nn.bias_add(y, biases)
    y_shape = y.get_shape().as_list()
    y = tf.reshape(y, [y_shape[0], y_shape[2], num_filters])
    y = batch_to_time(y, dilation)
    y.set_shape([batch_size, length, num_filters])
    return y


def pool1d(X, window_length, name, mode='avg', stride=None):
    """1D pooling function that supports multiple different modes.

    Parameters
    ----------
    X
        The [mb, time, channels] float tensor that we are going to pool over.
    window_length
        The amount of samples we pool over.
    name
        The name of the scope for the variables.
    mode
        The type of pooling, either avg or max.
    stride
        The stride length.

    Returns
    -------
    pooled
        The [mb, time // stride, channels] float tensor result of pooling.
    """
    if mode == 'avg':
        pool_fn = tf.nn.avg_pool
    elif mode == 'max':
        pool_fn = tf.nn.max_pool

    stride = stride or window_length
    batch_size, length, num_channels = X.get_shape().as_list()
    assert length % window_length == 0
    assert length % stride == 0

    window_shape = [1, 1, window_length, 1]
    strides = [1, 1, stride, 1]
    x_4d = tf.reshape(X, [batch_size, 1, length, num_channels])
    pooled = pool_fn(x_4d, window_shape, strides, padding='SAME', name=name)
    return tf.reshape(pooled, [batch_size, length // stride, num_channels])


def mu_law(X, mu=255, int8=False):
    """A TF implementation of Mu-Law encoding.

    Parameters
    ----------
    X
        The audio samples to encode.
    mu
        The Mu to use in our Mu-Law.
    int8
        Use int8 encoding.

    Returns
    -------
    out
        The Mu-Law encoded int8 data.
    """
    out = tf.sign(X) * tf.log(1 + mu * tf.abs(X)) / np.log(1 + mu)
    out = tf.floor(out * 128)
    if int8:
        out = tf.cast(out, tf.int8)
    return out


def mu_law_numpy(X, mu=255, int8=False):
    """A TF implementation of Mu-Law encoding.

    Parameters
    ----------
    X
        The audio samples to encode.
    mu
        The Mu to use in our Mu-Law.
    int8
        Use int8 encoding.

    Returns
    -------
    out
        The Mu-Law encoded int8 data.
    """
    out = np.sign(X) * np.log(1 + mu * np.abs(X)) / np.log(1 + mu)
    out = np.floor(out * 128)
    if int8:
        return out.astype(np.int8)
    return out


def inv_mu_law(X, mu=255):
    """A TF implementation of inverse Mu-Law.

    Parameters
    ----------
    X
        The Mu-Law samples to decode.
    mu
        The Mu we used to encode these samples.

    Returns
    -------
    out
        The decoded data.
    """
    X = tf.cast(X, tf.float32)
    out = (X + 0.5) * 2. / (mu + 1)
    out = tf.sign(out) / mu * ((1 + mu)**tf.abs(out) - 1)
    out = tf.where(tf.equal(X, 0), X, out)
    return out


def inv_mu_law_numpy(X, mu=255.0):
    """A numpy implementation of inverse Mu-Law.

    Parameters
    ----------
    X
        The Mu-Law samples to decode.
    mu
        The Mu we used to encode these samples.

    Returns
    -------
    out
        The decoded data.
    """
    X = np.array(X).astype(np.float32)
    out = (X + 0.5) * 2. / (mu + 1)
    out = np.sign(out) / mu * ((1 + mu)**np.abs(out) - 1)
    out = np.where(np.equal(X, 0), X, out)
    return out


def causal_linear(X, n_inputs, n_outputs, name, filter_length, rate,
                  batch_size, depth=1):
    """Applies dilated convolution using queues.
    Assumes a filter_length of 2 or 3.

    Parameters
    ----------
    X
        The [mb, time, channels] tensor input.
    n_inputs
        The input number of channels.
    n_outputs
        The output number of channels.
    name
        The variable scope to provide to W and biases.
    filter_length
        The length of the convolution, assumed to be 3.
    rate
        The rate or dilation
    batch_size
        Non-symbolic value for batch_size.
    depth : int, optional
        Description

    Returns
    -------
    y
        The output of the operation
    (init_1, init_2)
        Initialization operations for the queues
    (push_1, push_2)
        Push operations for the queues
    """
    assert filter_length == 2 or filter_length == 3

    # TODO: Make generic... started something like this:
    #    # create queue
    #    qs = []
    #    inits = []
    #    states = []
    #    pushs = []
    #    zeros = tf.zeros((rate, batch_size, depth, n_inputs))
    #    for f_i in range(1, filter_length):
    #        q = tf.FIFOQueue(
    #            rate,
    #            dtypes=tf.float32,
    #            shapes=(batch_size, depth, n_inputs))
    #        qs.append(q)
    #        inits.append(q.enqueue_many(zeros))
    #        states.append(q.dequeue())
    #
    #    pushs.append(qs[0].enqueue(X))
    #    for f_i in range(2, filter_length):
    #        pushs.append(qs[f_i].enqueue(states[f_i - 1]))

    if filter_length == 3:
        # create queue
        q_1 = tf.FIFOQueue(rate, dtypes=tf.float32, shapes=(batch_size, depth, n_inputs))
        q_2 = tf.FIFOQueue(rate, dtypes=tf.float32, shapes=(batch_size, depth, n_inputs))
        init_1 = q_1.enqueue_many(tf.zeros((rate, batch_size, depth, n_inputs)))
        init_2 = q_2.enqueue_many(tf.zeros((rate, batch_size, depth, n_inputs)))
        state_1 = q_1.dequeue()
        push_1 = q_1.enqueue(X)
        state_2 = q_2.dequeue()
        push_2 = q_2.enqueue(state_1)

        # get pretrained weights
        w = tf.get_variable(
            name=name + "/W",
            shape=[1, filter_length, n_inputs, n_outputs],
            dtype=tf.float32)
        b = tf.get_variable(
            name=name + "/biases", shape=[n_outputs], dtype=tf.float32)
        w_q_2 = tf.slice(w, [0, 0, 0, 0], [-1, 1, -1, -1])
        w_q_1 = tf.slice(w, [0, 1, 0, 0], [-1, 1, -1, -1])
        w_x = tf.slice(w, [0, 2, 0, 0], [-1, 1, -1, -1])

        # perform op w/ cached states
        y = tf.nn.bias_add(
            tf.matmul(state_2[:, 0, :], w_q_2[0][0]) + tf.matmul(
                state_1[:, 0, :], w_q_1[0][0]) + tf.matmul(X[:, 0, :], w_x[0][0]), b)

        y = tf.expand_dims(y, 1)
        return y, [init_1, init_2], [push_1, push_2]
    else:
        # create queue
        q = tf.FIFOQueue(
            rate,
            dtypes=tf.float32,
            shapes=(batch_size, depth, n_inputs))
        init = q.enqueue_many(
            tf.zeros((rate, batch_size, depth, n_inputs)))
        state = q.dequeue()
        push = q.enqueue(X)

        # get pretrained weights
        W = tf.get_variable(
            name=name + '/W',
            shape=[1, filter_length, n_inputs, n_outputs],
            dtype=tf.float32)
        b = tf.get_variable(
            name=name + '/biases',
            shape=[n_outputs],
            dtype=tf.float32)
        W_q = tf.slice(W, [0, 0, 0, 0], [-1, 1, -1, -1])
        W_x = tf.slice(W, [0, 1, 0, 0], [-1, 1, -1, -1])

        # perform op w/ cached states
        y = tf.nn.bias_add(
            tf.matmul(state[:, 0, :], W_q[0][0]) +
            tf.matmul(X[:, 0, :], W_x[0][0]),
            b)
        return tf.expand_dims(y, 1), [init], [push]


def linear(X, n_inputs, n_outputs, name):
    """Summary

    Parameters
    ----------
    X : TYPE
        Description
    n_inputs : TYPE
        Description
    n_outputs : TYPE
        Description
    name : TYPE
        Description

    Returns
    -------
    TYPE
        Description
    """
    W = tf.get_variable(
        name=name + '/W',
        shape=[1, 1, n_inputs, n_outputs],
        dtype=tf.float32)
    b = tf.get_variable(
        name=name + '/biases',
        shape=[n_outputs],
        dtype=tf.float32)
    # ipdb.set_trace()
    y = tf.nn.bias_add(tf.matmul(X[:, 0, :], W[0][0]), b)
    return tf.expand_dims(y, 1)
