"""NSynth: WaveNet Autoencoder.
"""
"""
NSynth model code and utilities are licensed under APL from the

Google Magenta project
----------------------
https://github.com/tensorflow/magenta/blob/master/magenta/models/nsynth

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
from scipy.io import wavfile
import numpy as np
from magenta.models.nsynth import utils
from magenta.models.nsynth import reader
from magenta.models.nsynth.wavenet import masked
from skimage.transform import resize


def get_model():
    """Summary
    """
    pass


def causal_linear(x, n_inputs, n_outputs, name, filter_length, rate, batch_size):
    """Summary

    Parameters
    ----------
    x : TYPE
        Description
    n_inputs : TYPE
        Description
    n_outputs : TYPE
        Description
    name : TYPE
        Description
    filter_length : TYPE
        Description
    rate : TYPE
        Description
    batch_size : TYPE
        Description

    Returns
    -------
    TYPE
        Description
    """
    # create queue
    q_1 = tf.FIFOQueue(
        rate,
        dtypes=tf.float32,
        shapes=(batch_size, n_inputs))
    q_2 = tf.FIFOQueue(
        rate,
        dtypes=tf.float32,
        shapes=(batch_size, n_inputs))
    init_1 = q_1.enqueue_many(
        tf.zeros((rate, batch_size, n_inputs)))
    init_2 = q_2.enqueue_many(
        tf.zeros((rate, batch_size, n_inputs)))
    state_1 = q_1.dequeue()
    push_1 = q_1.enqueue(x)
    state_2 = q_2.dequeue()
    push_2 = q_2.enqueue(state_1)

    # get pretrained weights
    W = tf.get_variable(
        name=name + '/W',
        shape=[1, filter_length, n_inputs, n_outputs],
        dtype=tf.float32)
    b = tf.get_variable(
        name=name + '/biases',
        shape=[n_outputs],
        dtype=tf.float32)
    W_q_2 = tf.slice(W, [0, 0, 0, 0], [-1, 1, -1, -1])
    W_q_1 = tf.slice(W, [0, 1, 0, 0], [-1, 1, -1, -1])
    W_x = tf.slice(W, [0, 2, 0, 0], [-1, 1, -1, -1])

    # perform op w/ cached states
    y = tf.expand_dims(tf.nn.bias_add(
        tf.matmul(state_2, W_q_2[0][0]) +
        tf.matmul(state_1, W_q_1[0][0]) +
        tf.matmul(x, W_x[0][0]),
        b), 0)
    return y, (init_1, init_2), (push_1, push_2)


def linear(x, n_inputs, n_outputs, name):
    """Summary

    Parameters
    ----------
    x : TYPE
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
    return tf.expand_dims(tf.nn.bias_add(tf.matmul(x[0], W[0][0]), b), 0)


class FastGenerationConfig(object):
    """Configuration object that helps manage the graph.
    """

    def __init__(self):
        """.
        """

    def build(self, inputs):
        """Build the graph for this configuration.

        Parameters
        ----------
        inputs
            A dict of inputs. For training, should contain 'wav'.

        Returns
        -------
        A dict of outputs that includes the 'predictions', 'loss', the 'encoding',
        the 'quantized_input', and whatever metrics we want to track for eval.

        Deleted Parameters
        ------------------
        is_training
            Whether we are training or not. Not used in this config.
        """
        num_stages = 10
        num_layers = 30
        filter_length = 3
        width = 512
        skip_width = 256
        num_z = 16

        # Encode the source with 8-bit Mu-Law.
        x = inputs['wav']
        batch_size = 1
        x_quantized = utils.mu_law(x)
        x_scaled = tf.cast(x_quantized, tf.float32) / 128.0
        x_scaled = tf.expand_dims(x_scaled, 2)

        encoding = tf.placeholder(
            name='encoding',
            shape=[num_z],
            dtype=tf.float32)
        en = tf.expand_dims(tf.expand_dims(encoding, 0), 0)

        init_ops, push_ops = [], []

        ###
        # The WaveNet Decoder.
        ###
        l = x_scaled
        l, inits, pushs = causal_linear(
            x=l[0],
            n_inputs=1,
            n_outputs=width,
            name='startconv',
            rate=1,
            batch_size=batch_size,
            filter_length=filter_length)
        [init_ops.append(init) for init in inits]
        [push_ops.append(push) for push in pushs]

        # Set up skip connections.
        s = linear(l, width, skip_width, name='skip_start')

        # Residual blocks with skip connections.
        for i in range(num_layers):
            dilation = 2**(i % num_stages)

            # dilated masked cnn
            d, inits, pushs = causal_linear(
                x=l[0],
                n_inputs=width,
                n_outputs=width * 2,
                name='dilatedconv_%d' % (i + 1),
                rate=dilation,
                batch_size=batch_size,
                filter_length=filter_length)
            [init_ops.append(init) for init in inits]
            [push_ops.append(push) for push in pushs]

            # local conditioning
            d = d + linear(en, num_z, width * 2, name='cond_map_%d' % (i + 1))

            # gated cnn
            assert d.get_shape().as_list()[2] % 2 == 0
            m = d.get_shape().as_list()[2] // 2
            d = tf.sigmoid(d[:, :, :m]) * tf.tanh(d[:, :, m:])

            # residuals
            l += linear(d, width, width, name='res_%d' % (i + 1))

            # skips
            s += linear(d, width, skip_width, name='skip_%d' % (i + 1))

        s = tf.nn.relu(s)
        s = linear(s, skip_width, skip_width, name='out1') + \
            linear(en, num_z, skip_width, name='cond_map_out1')
        s = tf.nn.relu(s)

        ###
        # Compute the logits and get the loss.
        ###
        logits = linear(s, skip_width, 256, name='logits')
        logits = tf.reshape(logits, [-1, 256])
        probs = tf.nn.softmax(logits, name='softmax')

        return {
            'init_ops': init_ops,
            'push_ops': push_ops,
            'encoding': encoding,
            'predictions': probs,
            'quantized_input': x_quantized,
        }


class Config(object):
    """Configuration object that helps manage the graph.

    Attributes
    ----------
    ae_bottleneck_width : int
        Description
    ae_hop_length : int
        Description
    encoding : TYPE
        Description
    learning_rate_schedule : TYPE
        Description
    num_iters : int
        Description
    train_path : TYPE
        Description
    """

    def __init__(self, encoding, train_path=None):
        """Summary

        Parameters
        ----------
        encoding : TYPE
            Description
        train_path : None, optional
            Description
        """
        self.num_iters = 200000
        self.learning_rate_schedule = {
            0: 2e-4,
            90000: 4e-4 / 3,
            120000: 6e-5,
            150000: 4e-5,
            180000: 2e-5,
            210000: 6e-6,
            240000: 2e-6,
        }
        self.ae_hop_length = 512
        self.ae_bottleneck_width = 16
        self.train_path = train_path
        self.encoding = encoding

    def get_batch(self, batch_size):
        """Summary

        Parameters
        ----------
        batch_size : TYPE
            Description

        Returns
        -------
        TYPE
            Description
        """
        assert self.train_path is not None
        data_train = reader.NSynthDataset(self.train_path, is_training=True)
        return data_train.get_wavenet_batch(batch_size, length=6144)

    @staticmethod
    def _condition(x, encoding):
        """Condition the input on the encoding.

        Parameters
        ----------
        x
            The [mb, length, channels] float tensor input.
        encoding
            The [mb, encoding_length, channels] float tensor encoding.

        Returns
        -------
        The output after broadcasting the encoding to x's shape and adding them.
        """
        mb, length, channels = x.get_shape().as_list()
        enc_mb, enc_length, enc_channels = encoding.get_shape().as_list()
        assert enc_mb == mb
        assert enc_channels == channels

        encoding = tf.reshape(encoding, [mb, enc_length, 1, channels])
        x = tf.reshape(x, [mb, enc_length, -1, channels])
        x += encoding
        x = tf.reshape(x, [mb, length, channels])
        x.set_shape([mb, length, channels])
        return x

    def build(self, inputs, is_training):
        """Build the graph for this configuration.

        Parameters
        ----------
        inputs
            A dict of inputs. For training, should contain 'wav'.
        is_training
            Whether we are training or not. Not used in this config.

        Returns
        -------
        A dict of outputs that includes the 'predictions', 'loss', the 'encoding',
        the 'quantized_input', and whatever metrics we want to track for eval.
        """
        del is_training
        num_stages = 10
        num_layers = 30
        filter_length = 3
        width = 512
        skip_width = 256
        ae_num_stages = 10
        ae_num_layers = 30
        ae_filter_length = 3
        ae_width = 128

        # Encode the source with 8-bit Mu-Law.
        x = inputs['wav']
        x_quantized = utils.mu_law(x)
        x_scaled = tf.cast(x_quantized, tf.float32) / 128.0
        x_scaled = tf.expand_dims(x_scaled, 2)

        if self.encoding:
            ###
            # The Non-Causal Temporal Encoder.
            ###
            en = masked.conv1d(
                x_scaled,
                causal=False,
                num_filters=ae_width,
                filter_length=ae_filter_length,
                name='ae_startconv')

            for num_layer in xrange(ae_num_layers):
                dilation = 2**(num_layer % ae_num_stages)
                d = tf.nn.relu(en)
                d = masked.conv1d(
                    d,
                    causal=False,
                    num_filters=ae_width,
                    filter_length=ae_filter_length,
                    dilation=dilation,
                    name='ae_dilatedconv_%d' % (num_layer + 1))
                d = tf.nn.relu(d)
                en += masked.conv1d(
                    d,
                    num_filters=ae_width,
                    filter_length=1,
                    name='ae_res_%d' % (num_layer + 1))

            en = masked.conv1d(
                en,
                num_filters=self.ae_bottleneck_width,
                filter_length=1,
                name='ae_bottleneck')
            en = masked.pool1d(en, self.ae_hop_length, name='ae_pool', mode='avg')
            encoding = en
        else:
            encoding = en = tf.placeholder(
                name='ae_pool',
                shape=[1, 125, 16],
                dtype=tf.float32)

        ###
        # The WaveNet Decoder.
        ###
        l = masked.shift_right(x_scaled)
        l = masked.conv1d(
            l, num_filters=width, filter_length=filter_length, name='startconv')

        # Set up skip connections.
        s = masked.conv1d(
            l, num_filters=skip_width, filter_length=1, name='skip_start')

        # Residual blocks with skip connections.
        for i in range(num_layers):
            dilation = 2**(i % num_stages)
            d = masked.conv1d(
                l,
                num_filters=2 * width,
                filter_length=filter_length,
                dilation=dilation,
                name='dilatedconv_%d' % (i + 1))
            d = self._condition(d,
                                masked.conv1d(
                                    en,
                                    num_filters=2 * width,
                                    filter_length=1,
                                    name='cond_map_%d' % (i + 1)))

            assert d.get_shape().as_list()[2] % 2 == 0
            m = d.get_shape().as_list()[2] // 2
            d_sigmoid = tf.sigmoid(d[:, :, :m])
            d_tanh = tf.tanh(d[:, :, m:])
            d = d_sigmoid * d_tanh

            l += masked.conv1d(
                d, num_filters=width, filter_length=1,
                name='res_%d' % (i + 1))
            s += masked.conv1d(
                d, num_filters=skip_width, filter_length=1,
                name='skip_%d' % (i + 1))

        s = tf.nn.relu(s)
        s = masked.conv1d(s, num_filters=skip_width, filter_length=1, name='out1')
        s = self._condition(s,
                            masked.conv1d(
                                en,
                                num_filters=skip_width,
                                filter_length=1,
                                name='cond_map_out1'))
        s = tf.nn.relu(s)

        ###
        # Compute the logits and get the loss.
        ###
        logits = masked.conv1d(s, num_filters=256, filter_length=1, name='logits')
        logits = tf.reshape(logits, [-1, 256])
        probs = tf.nn.softmax(logits, name='softmax')
        x_indices = tf.cast(tf.reshape(x_quantized, [-1]), tf.int32) + 128
        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits, labels=x_indices, name='nll'),
            0,
            name='loss')

        return {
            'predictions': probs,
            'loss': loss,
            'eval': {
                'nll': loss
            },
            'quantized_input': x_quantized,
            'encoding': encoding,
        }


def inv_mu_law(x, mu=255.0):
    """A TF implementation of inverse Mu-Law.

    Parameters
    ----------
    x
        The Mu-Law samples to decode.
    mu
        The Mu we used to encode these samples.

    Returns
    -------
    out
        The decoded data.
    """
    x = np.array(x).astype(np.float32)
    out = (x + 0.5) * 2. / (mu + 1)
    out = np.sign(out) / mu * ((1 + mu)**np.abs(out) - 1)
    out = np.where(np.equal(x, 0), x, out)
    return out


def load_audio(wav_file, sample_length=64000):
    """Summary

    Parameters
    ----------
    wav_file : TYPE
        Description
    sample_length : int, optional
        Description

    Returns
    -------
    TYPE
        Description
    """
    wav_data = np.array([utils.load_audio(wav_file)[:sample_length]])
    wav_data_padded = np.zeros((1, sample_length))
    wav_data_padded[0, :wav_data.shape[1]] = wav_data
    wav_data = wav_data_padded
    return wav_data


def load_nsynth(encoding=True, batch_size=1, sample_length=64000):
    """Summary

    Parameters
    ----------
    encoding : bool, optional
        Description
    batch_size : int, optional
        Description
    sample_length : int, optional
        Description

    Returns
    -------
    TYPE
        Description
    """
    config = Config(encoding=encoding)
    with tf.device('/gpu:0'):
        X = tf.placeholder(
            tf.float32, shape=[batch_size, sample_length])
        graph = config.build({"wav": X}, is_training=False)
        graph.update({'X': X})
    return graph


def load_fastgen_nsynth(batch_size=1, sample_length=64000):
    """Summary

    Parameters
    ----------
    batch_size : int, optional
        Description
    sample_length : int, optional
        Description

    Returns
    -------
    TYPE
        Description
    """
    config = FastGenerationConfig()
    X = tf.placeholder(
        tf.float32, shape=[batch_size, 1])
    graph = config.build({"wav": X})
    graph.update({'X': X})
    return graph


def synthesize(wav_file, out_file='synthesis.wav',
               sample_length=64000,
               synth_length=16000,
               ckpt_path='./model.ckpt-200000',
               resample_encoding=False):
    """Summary

    Parameters
    ----------
    wav_file : TYPE
        Description
    out_file : str, optional
        Description
    sample_length : int, optional
        Description
    synth_length : int, optional
        Description
    ckpt_path : str, optional
        Description
    resample_encoding : bool, optional
        Description

    Returns
    -------
    TYPE
        Description
    """
    # Audio to resynthesize
    wav_data = load_audio(wav_file, sample_length)

    # Load up the model for encoding and find the encoding of 'wav_data'
    with tf.Graph().as_default(), tf.Session() as sess:
        net = load_nsynth(encoding=True)
        saver = tf.train.Saver()
        saver.restore(sess, ckpt_path)
        encoding = sess.run(net['encoding'], feed_dict={
            net['X']: wav_data})[0]

    # Resample encoding to sample_length
    encoding_length = encoding.shape[0]
    if resample_encoding:
        max_val = np.max(np.abs(encoding))
        encoding = resize(encoding / max_val, (sample_length, 16))
        encoding = (encoding * max_val).astype(np.float32)

    with tf.Graph().as_default(), tf.Session() as sess:
        net = load_fastgen_nsynth()
        saver = tf.train.Saver()
        saver.restore(sess, ckpt_path)

        # initialize queues w/ 0s
        sess.run(net['init_ops'])

        # Regenerate the audio file sample by sample
        wav_synth = np.zeros((sample_length,),
                             dtype=np.float32)
        audio = np.float32(0)

        for sample_i in range(synth_length):
            print(sample_i)
            if resample_encoding:
                enc_i = sample_i
            else:
                enc_i = int(sample_i /
                            float(sample_length) *
                            float(encoding_length))
            res = sess.run(
                [net['predictions'], net['push_ops']],
                feed_dict={
                    net['X']: np.atleast_2d(audio),
                    net['encoding']: encoding[enc_i]})[0]
            cdf = np.cumsum(res)
            idx = np.random.rand()
            i = 0
            while(cdf[i] < idx):
                i = i + 1
            audio = inv_mu_law(i - 128)
            wav_synth[sample_i] = audio

    wavfile.write(out_file, 16000, wav_synth)

    sess.close()
    return wav_synth
