"""WaveNet Training and Fast WaveNet Decoding.

From the following paper
------------------------
Ramachandran, P., Le Paine, T., Khorrami, P., Babaeizadeh, M., Chang, S.,
Zhang, Y., … Huang, T. (2017). Fast Generation For Convolutional
Autoregressive Models, 1–5.
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
import numpy as np
import tensorflow as tf
from cadl import librispeech
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


def create_generation_model(n_stages=5, n_layers_per_stage=10,
                            n_hidden=256, batch_size=1, n_skip=128,
                            n_quantization=256, filter_length=2,
                            onehot=False):
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
    n_quantization : int, optional
        Description
    filter_length : int, optional
        Description
    onehot : bool, optional
        Description

    Returns
    -------
    TYPE
        Description
    """
    offset = n_quantization / 2.0

    # Encode the source with 8-bit Mu-Law.
    X = tf.placeholder(name='X', shape=[None, None], dtype=tf.float32)
    X_quantized = wnu.mu_law(X, n_quantization)
    if onehot:
        X_onehot = tf.one_hot(
            tf.cast(X_quantized + offset, tf.int32),
            n_quantization)
    else:
        X_onehot = tf.expand_dims(X_quantized, 2)

    push_ops, init_ops = [], []
    h, init, push = wnu.causal_linear(
        X=X_onehot,
        n_inputs=256 if onehot else 1,
        n_outputs=n_hidden,
        name='startconv',
        rate=1,
        batch_size=batch_size,
        filter_length=filter_length)
    init_ops.extend(init)
    push_ops.extend(push)

    # Set up skip connections.
    s = wnu.linear(h, n_hidden, n_skip, name='skip_start')

    # Residual blocks with skip connections.
    for i in range(n_stages * n_layers_per_stage):
        dilation = 2**(i % n_layers_per_stage)

        # dilated masked cnn
        d, init, push = wnu.causal_linear(
            X=h,
            n_inputs=n_hidden,
            n_outputs=n_hidden * 2,
            name='dilatedconv_%d' % (i + 1),
            rate=dilation,
            batch_size=batch_size,
            filter_length=filter_length)
        init_ops.extend(init)
        push_ops.extend(push)

        # gated cnn
        assert d.get_shape().as_list()[2] % 2 == 0
        m = d.get_shape().as_list()[2] // 2
        d = tf.sigmoid(d[:, :, :m]) * tf.tanh(d[:, :, m:])

        # residuals
        h += wnu.linear(d, n_hidden, n_hidden, name='res_%d' % (i + 1))

        # skips
        s += wnu.linear(d, n_hidden, n_skip, name='skip_%d' % (i + 1))

    s = tf.nn.relu(s)
    s = wnu.linear(s, n_skip, n_skip, name='out1')
    s = tf.nn.relu(s)
    logits = tf.clip_by_value(
        wnu.linear(s, n_skip, n_quantization, name='logits_preclip') + offset,
        0.0, n_quantization - 1.0,
        name='logits')
    logits = tf.reshape(logits, [-1, n_quantization])
    probs = tf.nn.softmax(logits, name='softmax')
    synthesis = tf.reshape(
        wnu.inv_mu_law(tf.cast(tf.argmax(probs, 1), tf.float32) - offset,
                       n_quantization),
        [-1, 1])

    return {
        'X': X,
        'init_ops': init_ops,
        'push_ops': push_ops,
        'probs': probs,
        'synthesis': synthesis
    }


def test_librispeech():
    """Summary
    """
    prime_length = 6144
    total_length = 16000 * 3
    batch_size = 32
    n_stages = 6
    n_layers_per_stage = 9
    n_hidden = 32
    filter_length = 2
    n_skip = 256
    onehot = False

    sequence_length = get_sequence_length(n_stages, n_layers_per_stage)
    ckpt_path = 'vctk-wavenet/wavenet_filterlen{}_batchsize{}_sequencelen{}_stages{}_layers{}_hidden{}_skips{}/'.format(
        filter_length, batch_size, sequence_length,
        n_stages, n_layers_per_stage, n_hidden, n_skip)

    dataset = librispeech.get_dataset()
    batch = next(librispeech.batch_generator(dataset,
                                             batch_size, prime_length))[0]

    with tf.Graph().as_default(), tf.Session() as sess:
        net = create_generation_model(batch_size=batch_size,
                                      filter_length=filter_length,
                                      n_hidden=n_hidden,
                                      n_skip=n_skip,
                                      n_layers_per_stage=n_layers_per_stage,
                                      n_stages=n_stages,
                                      onehot=onehot)
        saver = tf.train.Saver()
        if tf.train.latest_checkpoint(ckpt_path) is not None:
            saver.restore(sess, tf.train.latest_checkpoint(ckpt_path))
        else:
            print('Could not find checkpoint')
        sess.run(net['init_ops'])
        synth = np.zeros([batch_size, total_length], dtype=np.float32)
        synth[:, :prime_length] = batch

        print('Synthesize...')
        for sample_i in range(total_length - 1):
            print('{}/{}/{}'.format(sample_i, prime_length, total_length),
                  end='\r')
            probs = sess.run(
                [net["probs"], net["push_ops"]],
                feed_dict={net["X"]: synth[:, [sample_i]]})[0]
            idxs = sample_categorical(probs)
            audio = wnu.inv_mu_law_numpy(idxs - 128)
            if sample_i >= prime_length:
                synth[:, sample_i + 1] = audio

        for i in range(batch_size):
            wavfile.write('synthesis-{}.wav'.format(i),
                          16000, synth[i])
