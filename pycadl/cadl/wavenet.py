"""WaveNet Autoencoder and conditional WaveNet.
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
import os
import numpy as np
import tensorflow as tf
from cadl import librispeech, vctk
from cadl import wavenet_utils as wnu
from cadl.utils import sample_categorical
from scipy.io import wavfile


def get_sequence_length(n_stages, n_layers_per_stage):
    """Summary

    Parameters
    ----------
    n_stages : TYPE
        Description
    n_layers_per_stage : TYPE
        Description

    Returns
    -------
    TYPE
        Description
    """
    sequence_length = 2**n_layers_per_stage * 2 * n_stages
    return sequence_length


def condition(x, encoding):
    """Summary

    Parameters
    ----------
    x : TYPE
        Description
    encoding : TYPE
        Description

    Returns
    -------
    TYPE
        Description
    """
    batch_size, length, channels = x.get_shape().as_list()
    enc_batch_size, enc_length, enc_channels = encoding.get_shape().as_list()
    assert enc_batch_size == batch_size
    assert enc_channels == channels
    encoding = tf.reshape(encoding, [batch_size, enc_length, 1, channels])
    x = tf.reshape(x, [batch_size, enc_length, -1, channels])
    x += encoding
    x = tf.reshape(x, [batch_size, length, channels])
    x.set_shape([batch_size, length, channels])
    return x


def create_wavenet_autoencoder(n_stages, n_layers_per_stage, n_hidden,
                               batch_size, n_skip, filter_length,
                               bottleneck_width, hop_length, n_quantization,
                               sample_rate):
    """Summary

    Parameters
    ----------
    n_stages : TYPE
        Description
    n_layers_per_stage : TYPE
        Description
    n_hidden : TYPE
        Description
    batch_size : TYPE
        Description
    n_skip : TYPE
        Description
    filter_length : TYPE
        Description
    bottleneck_width : TYPE
        Description
    hop_length : TYPE
        Description
    n_quantization : TYPE
        Description
    sample_rate : TYPE
        Description

    Returns
    -------
    TYPE
        Description
    """
    offset = n_quantization / 2.0
    sequence_length = 2**n_layers_per_stage * 2 * n_stages

    # Encode the source with 8-bit Mu-Law.
    X = tf.placeholder(
        name='X', shape=[batch_size, sequence_length], dtype=tf.float32)
    X_quantized = wnu.mu_law(X, n_quantization)
    X_scaled = tf.cast(X_quantized / offset, tf.float32)
    X_scaled = tf.expand_dims(X_scaled, 2)

    # The Non-Causal Temporal Encoder.
    en = wnu.conv1d(
        X=X_scaled,
        causal=False,
        num_filters=n_hidden,
        filter_length=filter_length,
        name='ae_startconv')

    # Residual blocks with skip connections.
    for i in range(n_stages * n_layers_per_stage):
        dilation = 2**(i % n_layers_per_stage)
        print(dilation)
        d = tf.nn.relu(en)
        d = wnu.conv1d(
            d,
            causal=False,
            num_filters=n_hidden,
            filter_length=filter_length,
            dilation=dilation,
            name='ae_dilatedconv_%d' % (i + 1))
        d = tf.nn.relu(d)
        en += wnu.conv1d(
            d,
            num_filters=n_hidden,
            filter_length=1,
            name='ae_res_%d' % (i + 1))

    en = wnu.conv1d(
        en, num_filters=bottleneck_width, filter_length=1, name='ae_bottleneck')

    en = wnu.pool1d(en, hop_length, name='ae_pool', mode='avg')
    encoding = en

    # The WaveNet Decoder.
    l = wnu.shift_right(X_scaled)
    l = wnu.conv1d(
        l, num_filters=n_hidden, filter_length=filter_length, name='startconv')

    # Set up skip connections.
    s = wnu.conv1d(l, num_filters=n_skip, filter_length=1, name='skip_start')

    # Residual blocks with skip connections.
    for i in range(n_stages * n_layers_per_stage):
        dilation = 2**(i % n_layers_per_stage)
        d = wnu.conv1d(
            l,
            num_filters=2 * n_hidden,
            filter_length=filter_length,
            dilation=dilation,
            name='dilatedconv_%d' % (i + 1))
        d = condition(d,
                      wnu.conv1d(
                          en,
                          num_filters=2 * n_hidden,
                          filter_length=1,
                          name='cond_map_%d' % (i + 1)))
        assert d.get_shape().as_list()[2] % 2 == 0
        m = d.get_shape().as_list()[2] // 2
        d_sigmoid = tf.sigmoid(d[:, :, :m])
        d_tanh = tf.tanh(d[:, :, m:])
        d = d_sigmoid * d_tanh
        l += wnu.conv1d(
            d, num_filters=n_hidden, filter_length=1, name='res_%d' % (i + 1))
        s += wnu.conv1d(
            d, num_filters=n_skip, filter_length=1, name='skip_%d' % (i + 1))

    s = tf.nn.relu(s)
    s = wnu.conv1d(s, num_filters=n_skip, filter_length=1, name='out1')
    s = condition(s,
                  wnu.conv1d(
                      en,
                      num_filters=n_skip,
                      filter_length=1,
                      name='cond_map_out1'))
    s = tf.nn.relu(s)

    # Compute the logits and get the loss.
    logits = wnu.conv1d(
        s, num_filters=n_quantization, filter_length=1, name='logits')
    logits = tf.reshape(logits, [-1, n_quantization])
    probs = tf.nn.softmax(logits, name='softmax')
    synthesis = tf.reshape(
        wnu.inv_mu_law(
            tf.cast(tf.argmax(probs, 1), tf.float32) - offset, n_quantization),
        [-1, sequence_length])
    labels = tf.cast(tf.reshape(X_quantized, [-1]), tf.int32) + int(offset)
    loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=labels, name='nll'),
        0,
        name='loss')

    tf.summary.audio("synthesis", synthesis, sample_rate=sample_rate)
    tf.summary.histogram("probs", probs)
    tf.summary.histogram("input_quantized", X_quantized)
    tf.summary.histogram("logits", logits)
    tf.summary.histogram("labels", labels)
    tf.summary.histogram("synthesis", synthesis)
    tf.summary.scalar("loss", loss)
    summaries = tf.summary.merge_all()

    return {
        'X': X,
        'quantized': X_quantized,
        'encoding': encoding,
        'probs': probs,
        'synthesis': synthesis,
        'summaries': summaries,
        'loss': loss
    }


def create_wavenet(n_stages=10,
                   n_layers_per_stage=9,
                   n_hidden=200,
                   batch_size=32,
                   n_skip=100,
                   filter_length=2,
                   shift=True,
                   n_quantization=256,
                   sample_rate=16000):
    """Summary

    Parameters
    ----------
    n_stages : int, optional
        Description
    n_layers_per_stage : int, optional
        Description
    n_hidden : int, optional
        Description
    batch_size : int, optional
        Description
    n_skip : int, optional
        Description
    filter_length : int, optional
        Description
    shift : bool, optional
        Description
    n_quantization : int, optional
        Description
    sample_rate : int, optional
        Description

    Returns
    -------
    TYPE
        Description
    """
    offset = n_quantization / 2.0
    sequence_length = 2**n_layers_per_stage * 2 * n_stages

    # Encode the source with 8-bit Mu-Law.
    X = tf.placeholder(
        name='X', shape=[batch_size, sequence_length], dtype=tf.float32)
    X_quantized = wnu.mu_law(X, n_quantization)
    X_onehot = tf.expand_dims(X_quantized, 2)
    if shift:
        X_onehot = wnu.shift_right(X_onehot)

    h = wnu.conv1d(
        X=X_onehot,
        num_filters=n_hidden,
        filter_length=filter_length,
        name='startconv')

    # Set up skip connections.
    s = wnu.conv1d(X=h, num_filters=n_skip, filter_length=1, name='skip_start')

    # Residual blocks with skip connections.
    for i in range(n_stages * n_layers_per_stage):
        dilation = 2**(i % n_layers_per_stage)

        # dilated masked cnn
        d = wnu.conv1d(
            X=h,
            num_filters=2 * n_hidden,
            filter_length=filter_length,
            dilation=dilation,
            name='dilatedconv_%d' % (i + 1))

        # gated cnn
        assert d.get_shape().as_list()[2] % 2 == 0
        m = d.get_shape().as_list()[2] // 2
        d = tf.sigmoid(d[:, :, :m]) * tf.tanh(d[:, :, m:])

        # residuals
        h += wnu.conv1d(
            X=d, num_filters=n_hidden, filter_length=1, name='res_%d' % (i + 1))

        # skips
        s += wnu.conv1d(
            X=d, num_filters=n_skip, filter_length=1, name='skip_%d' % (i + 1))

    s = tf.nn.relu(s)
    s = wnu.conv1d(X=s, num_filters=n_skip, filter_length=1, name='out1')
    s = tf.nn.relu(s)
    logits = tf.clip_by_value(
        wnu.conv1d(
            X=s,
            num_filters=n_quantization,
            filter_length=1,
            name='logits_preclip') + offset,
        0.0,
        n_quantization - 1.0,
        name='logits')
    logits = tf.reshape(logits, [-1, n_quantization])
    labels = tf.cast(tf.reshape(X_quantized + offset, [-1]), tf.int32)
    loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=labels, name='nll'),
        0,
        name='loss')

    probs = tf.nn.softmax(logits, name='softmax')
    synthesis = tf.reshape(
        wnu.inv_mu_law(
            tf.cast(tf.argmax(probs, 1), tf.float32) - offset, n_quantization),
        [-1, sequence_length])

    tf.summary.audio("synthesis", synthesis, sample_rate=sample_rate)
    tf.summary.histogram("probs", probs)
    tf.summary.histogram("input_quantized", X_quantized)
    tf.summary.histogram("logits", logits)
    tf.summary.histogram("labels", labels)
    tf.summary.histogram("synthesis", synthesis)
    tf.summary.scalar("loss", loss)
    summaries = tf.summary.merge_all()

    return {
        'X': X,
        'quantized': X_quantized,
        'probs': probs,
        'synthesis': synthesis,
        'summaries': summaries,
        'loss': loss
    }


def train_vctk():
    """Summary

    Returns
    -------
    TYPE
        Description
    """
    batch_size = 24
    filter_length = 2
    n_stages = 7
    n_layers_per_stage = 9
    n_hidden = 48
    n_skip = 384
    dataset = vctk.get_dataset()
    it_i = 0
    n_epochs = 1000
    sequence_length = get_sequence_length(n_stages, n_layers_per_stage)
    ckpt_path = 'vctk-wavenet/wavenet_filterlen{}_batchsize{}_sequencelen{}_stages{}_layers{}_hidden{}_skips{}'.format(
        filter_length, batch_size, sequence_length, n_stages,
        n_layers_per_stage, n_hidden, n_skip)
    with tf.graph().as_default(), tf.session() as sess:
        net = create_wavenet(
            batch_size=batch_size,
            filter_length=filter_length,
            n_hidden=n_hidden,
            n_skip=n_skip,
            n_stages=n_stages,
            n_layers_per_stage=n_layers_per_stage)
        saver = tf.train.saver()
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        sess.run(init_op)
        if tf.train.latest_checkpoint(ckpt_path) is not None:
            saver.restore(sess, tf.train.latest_checkpoint(ckpt_path))
        batch = vctk.batch_generator
        with tf.variable_scope('optimizer'):
            opt = tf.train.adamoptimizer(
                learning_rate=0.0002).minimize(net['loss'])
        var_list = [
            v for v in tf.global_variables() if v.name.startswith('optimizer')
        ]
        sess.run(tf.variables_initializer(var_list))
        writer = tf.summary.filewriter(ckpt_path)
        for epoch_i in range(n_epochs):
            for batch_xs in batch(dataset, batch_size, sequence_length):
                loss, quantized, _ = sess.run(
                    [net['loss'], net['quantized'], opt],
                    feed_dict={net['x']: batch_xs})
                print(loss)
                if it_i % 100 == 0:
                    summary = sess.run(
                        net['summaries'], feed_dict={net['x']: batch_xs})
                    writer.add_summary(summary, it_i)
                    # save
                    saver.save(
                        sess,
                        os.path.join(ckpt_path, 'model.ckpt'),
                        global_step=it_i)
                it_i += 1

    return loss


def test_librispeech():
    """Summary
    """
    batch_size = 24
    filter_length = 2
    n_stages = 7
    n_layers_per_stage = 9
    n_hidden = 48
    n_skip = 384
    total_length = 16000
    sequence_length = get_sequence_length(n_stages, n_layers_per_stage)
    prime_length = sequence_length
    ckpt_path = 'wavenet/wavenet_filterlen{}_batchsize{}_sequencelen{}_stages{}_layers{}_hidden{}_skips{}/'.format(
        filter_length, batch_size, sequence_length, n_stages,
        n_layers_per_stage, n_hidden, n_skip)

    dataset = librispeech.get_dataset()
    batch = next(
        librispeech.batch_generator(dataset, batch_size, prime_length))[0]

    sess = tf.Session()
    net = create_wavenet(
        batch_size=batch_size,
        filter_length=filter_length,
        n_hidden=n_hidden,
        n_skip=n_skip,
        n_layers_per_stage=n_layers_per_stage,
        n_stages=n_stages,
        shift=False)
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    sess.run(init_op)
    saver = tf.train.Saver()
    if tf.train.latest_checkpoint(ckpt_path) is not None:
        saver.restore(sess, tf.train.latest_checkpoint(ckpt_path))
    else:
        print('Could not find checkpoint')

    synth = np.zeros([batch_size, total_length], dtype=np.float32)
    synth[:, :prime_length] = batch

    print('Synthesize...')
    for sample_i in range(0, total_length - prime_length):
        print('{}/{}/{}'.format(sample_i, prime_length, total_length), end='\r')
        probs = sess.run(
            net["probs"],
            feed_dict={net["X"]: synth[:, sample_i:sample_i + sequence_length]})
        idxs = sample_categorical(probs)
        idxs = idxs.reshape((batch_size, sequence_length))
        if sample_i == 0:
            audio = wnu.inv_mu_law_numpy(idxs - 128)
            synth[:, :prime_length] = audio
        else:
            audio = wnu.inv_mu_law_numpy(idxs[:, -1] - 128)
            synth[:, prime_length + sample_i] = audio

    for i in range(batch_size):
        wavfile.write('synthesis-{}.wav'.format(i), 16000, synth[i])
