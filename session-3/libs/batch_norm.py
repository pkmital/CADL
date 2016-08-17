"""Batch Normalization for TensorFlow.
Parag K. Mital, Jan 2016.
"""

import tensorflow as tf
from tensorflow.python import control_flow_ops


def batch_norm(x, phase_train, name='bn', decay=0.99, reuse=None, affine=True):
    """
    Batch normalization on convolutional maps.
    from: https://stackoverflow.com/questions/33949786/how-could-i-
    use-batch-normalization-in-tensorflow
    Only modified to infer shape from input tensor x.
    Parameters
    ----------
    x
        Tensor, 4D BHWD input maps
    phase_train
        boolean tf.Variable, true indicates training phase
    name
        string, variable name
    affine
        whether to affine-transform outputs
    Return
    ------
    normed
        batch-normalized maps
    """
    with tf.variable_scope(name, reuse=reuse):
        og_shape = x.get_shape().as_list()
        if len(og_shape) == 2:
            x = tf.reshape(x, [-1, 1, 1, og_shape[1]])
        shape = x.get_shape().as_list()
        beta = tf.get_variable(name='beta', shape=[shape[-1]],
                               initializer=tf.constant_initializer(0.0),
                               trainable=True)
        gamma = tf.get_variable(name='gamma', shape=[shape[-1]],
                                initializer=tf.constant_initializer(1.0),
                                trainable=affine)

        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=decay)
        ema_apply_op = ema.apply([batch_mean, batch_var])
        ema_mean, ema_var = ema.average(batch_mean), ema.average(batch_var)

        def mean_var_with_update():
            """Summary
            Returns
            -------
            name : TYPE
                Description
            """
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)
        mean, var = control_flow_ops.cond(phase_train,
                                          mean_var_with_update,
                                          lambda: (ema_mean, ema_var))

        # tf.nn.batch_normalization
        normed = tf.nn.batch_norm_with_global_normalization(
            x, mean, var, beta, gamma, 1e-5, affine)
        if len(og_shape) == 2:
            normed = tf.reshape(normed, [-1, og_shape[-1]])
    return normed
