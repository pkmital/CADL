"""Batch Normalization for TensorFlow.
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
from tensorflow.python.ops import control_flow_ops


def batch_norm(x, phase_train, name='bn', decay=0.9, reuse=None, affine=True):
    """
    Batch normalization on convolutional maps.
    from: https://stackoverflow.com/questions/33949786/how-could-i-
    use-batch-normalization-in-tensorflow
    Only modified to infer shape from input tensor x.

    [DEPRECATED] Use tflearn or slim batch normalization instead.

    Parameters
    ----------
    x
        Tensor, 4D BHWD input maps
    phase_train
        boolean tf.Variable, true indicates training phase
    name
        string, variable name
    decay : float, optional
        Description
    reuse : None, optional
        Description
    affine
        whether to affine-transform outputs

    Return
    ------
    normed
        batch-normalized maps
    """
    with tf.variable_scope(name, reuse=reuse):
        shape = x.get_shape().as_list()
        beta = tf.get_variable(
            name='beta',
            shape=[shape[-1]],
            initializer=tf.constant_initializer(0.0),
            trainable=True)
        gamma = tf.get_variable(
            name='gamma',
            shape=[shape[-1]],
            initializer=tf.constant_initializer(1.0),
            trainable=affine)
        if len(shape) == 4:
            batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
        else:
            batch_mean, batch_var = tf.nn.moments(x, [0], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=decay)
        ema_apply_op = ema.apply([batch_mean, batch_var])
        ema_mean, ema_var = ema.average(batch_mean), ema.average(batch_var)

        def mean_var_with_update():
           with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = control_flow_ops.cond(phase_train, mean_var_with_update,
                                          lambda: (ema_mean, ema_var))

        # tf.nn.batch_normalization
        normed = tf.nn.batch_norm_with_global_normalization(
            x, mean, var, beta, gamma, 1e-6, affine)
    return normed
