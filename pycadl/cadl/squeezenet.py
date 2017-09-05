"""SqueezeNet
"""
"""
squeezeNet is a much smaller convolutional network, with vastly less amount of parameters.
queezeNet achieves AlexNet-level accuracy on ImageNet with 50x fewer parameters.
Additionally, with model compression techniques SqueezeNet can be compress to less than 0.5MB (510x smaller than AlexNet).

http://arxiv.org/abs/1602.07360

Code taken from https://github.com/Khushmeet/squeezeNet/blob/master/squeezenet/squeezenet.py

MIT License

Copyright (c) 2017 Khushmeet

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

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


def fire_module(
        input,
        fire_id,
        channel,
        s1,
        e1,
        e3, ):
    """
    Basic module that makes up the SqueezeNet architecture. It has two layers.
     1. Squeeze layer (1x1 convolutions)
     2. Expand layer (1x1 and 3x3 convolutions)
    :param input: Tensorflow tensor
    :param fire_id: Variable scope name
    :param channel: Depth of the previous output
    :param s1: Number of filters for squeeze 1x1 layer
    :param e1: Number of filters for expand 1x1 layer
    :param e3: Number of filters for expand 3x3 layer
    :return: Tensorflow tensor

    Parameters
    ----------
    input : TYPE
        Description
    fire_id : TYPE
        Description
    channel : TYPE
        Description
    s1 : TYPE
        Description
    e1 : TYPE
        Description
    e3 : TYPE
        Description

    Returns
    -------
    TYPE
        Description
    """

    fire_weights = {
        'conv_s_1': tf.Variable(tf.truncated_normal([1, 1, channel, s1])),
        'conv_e_1': tf.Variable(tf.truncated_normal([1, 1, s1, e1])),
        'conv_e_3': tf.Variable(tf.truncated_normal([3, 3, s1, e3]))
    }

    fire_biases = {
        'conv_s_1': tf.Variable(tf.truncated_normal([s1])),
        'conv_e_1': tf.Variable(tf.truncated_normal([e1])),
        'conv_e_3': tf.Variable(tf.truncated_normal([e3]))
    }

    with tf.name_scope(fire_id):
        output = tf.nn.conv2d(
            input,
            fire_weights['conv_s_1'],
            strides=[1, 1, 1, 1],
            padding='SAME',
            name='conv_s_1')
        output = tf.nn.relu(tf.nn.bias_add(output, fire_biases['conv_s_1']))

        expand1 = tf.nn.conv2d(
            output,
            fire_weights['conv_e_1'],
            strides=[1, 1, 1, 1],
            padding='SAME',
            name='conv_e_1')
        expand1 = tf.nn.bias_add(expand1, fire_biases['conv_e_1'])

        expand3 = tf.nn.conv2d(
            output,
            fire_weights['conv_e_3'],
            strides=[1, 1, 1, 1],
            padding='SAME',
            name='conv_e_3')
        expand3 = tf.nn.bias_add(expand3, fire_biases['conv_e_3'])

        result = tf.concat([expand1, expand3], 3, name='concat_e1_e3')
        return tf.nn.relu(result)


def squeeze_net(input, classes):
    """
    SqueezeNet model written in tensorflow. It provides AlexNet level accuracy with 50x fewer parameters
    and smaller model size.
    :param input: Input tensor (4D)
    :param classes: number of classes for classification
    :return: Tensorflow tensor

    Parameters
    ----------
    input : TYPE
        Description
    classes : TYPE
        Description

    Returns
    -------
    TYPE
        Description
    """

    weights = {
        'conv1': tf.Variable(tf.truncated_normal([7, 7, 1, 96])),
        'conv10': tf.Variable(tf.truncated_normal([1, 1, 512, classes]))
    }

    biases = {
        'conv1': tf.Variable(tf.truncated_normal([96])),
        'conv10': tf.Variable(tf.truncated_normal([classes]))
    }

    output = tf.nn.conv2d(
        input,
        weights['conv1'],
        strides=[1, 2, 2, 1],
        padding='SAME',
        name='conv1')
    output = tf.nn.bias_add(output, biases['conv1'])

    output = tf.nn.max_pool(
        output,
        ksize=[1, 3, 3, 1],
        strides=[1, 2, 2, 1],
        padding='SAME',
        name='maxpool1')

    output = fire_module(
        output, s1=16, e1=64, e3=64, channel=96, fire_id='fire2')
    output = fire_module(
        output, s1=16, e1=64, e3=64, channel=128, fire_id='fire3')
    output = fire_module(
        output, s1=32, e1=128, e3=128, channel=128, fire_id='fire4')

    output = tf.nn.max_pool(
        output,
        ksize=[1, 3, 3, 1],
        strides=[1, 2, 2, 1],
        padding='SAME',
        name='maxpool4')

    output = fire_module(
        output, s1=32, e1=128, e3=128, channel=256, fire_id='fire5')
    output = fire_module(
        output, s1=48, e1=192, e3=192, channel=256, fire_id='fire6')
    output = fire_module(
        output, s1=48, e1=192, e3=192, channel=384, fire_id='fire7')
    output = fire_module(
        output, s1=64, e1=256, e3=256, channel=384, fire_id='fire8')

    output = tf.nn.max_pool(
        output,
        ksize=[1, 3, 3, 1],
        strides=[1, 2, 2, 1],
        padding='SAME',
        name='maxpool8')

    output = fire_module(
        output, s1=64, e1=256, e3=256, channel=512, fire_id='fire9')

    output = tf.nn.dropout(output, keep_prob=0.5, name='dropout9')

    output = tf.nn.conv2d(
        output,
        weights['conv10'],
        strides=[1, 1, 1, 1],
        padding='SAME',
        name='conv10')
    output = tf.nn.bias_add(output, biases['conv10'])

    output = tf.nn.avg_pool(
        output,
        ksize=[1, 13, 13, 1],
        strides=[1, 2, 2, 1],
        padding='SAME',
        name='avgpool10')

    return output
