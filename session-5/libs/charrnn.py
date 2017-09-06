"""Character-level Recurrent Neural Network.
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
import numpy as np
import os
import sys
import collections
import gzip
from libs import utils


def build_model(txt,
                batch_size=1,
                sequence_length=1,
                n_layers=2,
                n_cells=100,
                gradient_clip=10.0,
                learning_rate=0.001):
    """Summary

    Parameters
    ----------
    txt : TYPE
        Description
    batch_size : int, optional
        Description
    sequence_length : int, optional
        Description
    n_layers : int, optional
        Description
    n_cells : int, optional
        Description
    gradient_clip : float, optional
        Description
    learning_rate : float, optional
        Description

    Returns
    -------
    TYPE
        Description
    """
    vocab = list(set(txt))
    vocab.sort()
    n_chars = len(vocab)
    encoder = collections.OrderedDict(zip(vocab, range(n_chars)))
    decoder = collections.OrderedDict(zip(range(n_chars), vocab))

    X = tf.placeholder(tf.int32, [None, sequence_length], name='X')
    Y = tf.placeholder(tf.int32, [None, sequence_length], name='Y')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    with tf.variable_scope('embedding'):
        embedding = tf.get_variable("embedding", [n_chars, n_cells])
        # Each sequence element will be connected to n_cells
        Xs = tf.nn.embedding_lookup(embedding, X)
        # Then slice each sequence element, giving us sequence number of
        # batch x 1 x n_chars Tensors
        Xs = tf.split(axis=1, num_or_size_splits=sequence_length, value=Xs)
        # Get rid of singleton sequence element dimension
        Xs = [tf.squeeze(X_i, [1]) for X_i in Xs]

    with tf.variable_scope('rnn'):
        cells = tf.contrib.rnn.MultiRNNCell([
            tf.contrib.rnn.DropoutWrapper(
                tf.contrib.rnn.BasicLSTMCell(
                    num_units=n_cells, forget_bias=0.0, state_is_tuple=True),
                output_keep_prob=keep_prob) for _ in range(n_layers)
        ])
        initial_state = cells.zero_state(tf.shape(X)[0], tf.float32)
        # returns a length sequence length list of outputs, one for each input
        outputs, final_state = tf.contrib.rnn.static_rnn(
            cells, Xs, initial_state=initial_state)
        # now concat the sequence length number of batch x n_cells Tensors to
        # give [sequence_length x batch, n_cells]
        outputs_flat = tf.reshape(
            tf.concat(axis=1, values=outputs), [-1, n_cells])

    with tf.variable_scope('prediction'):
        W = tf.get_variable(
            "W",
            shape=[n_cells, n_chars],
            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable(
            "b", shape=[n_chars], initializer=tf.constant_initializer())
        logits = tf.matmul(outputs_flat, W) + b
        probs = tf.nn.softmax(logits)
        Y_pred = tf.argmax(probs, 1)

    with tf.variable_scope('loss'):
        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example([logits], [
            tf.reshape(tf.concat(axis=1, values=Y), [-1])
        ], [tf.ones([batch_size * sequence_length])])
        cost = tf.reduce_sum(loss) / batch_size

    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        gradients = []
        clip = tf.constant(gradient_clip, name="clip")
        for grad, var in optimizer.compute_gradients(cost):
            gradients.append((tf.clip_by_value(grad, -clip, clip), var))
        updates = optimizer.apply_gradients(gradients)

    model = {
        'X': X,
        'Y': Y,
        'logits': logits,
        'probs': probs,
        'Y_pred': Y_pred,
        'keep_prob': keep_prob,
        'cost': cost,
        'updates': updates,
        'initial_state': initial_state,
        'final_state': final_state,
        'decoder': decoder,
        'encoder': encoder,
        'vocab_size': n_chars
    }
    return model


def train(txt,
          batch_size=100,
          sequence_length=150,
          n_cells=200,
          n_layers=3,
          learning_rate=0.00001,
          max_iter=50000,
          gradient_clip=5.0,
          ckpt_name="model.ckpt",
          keep_prob=1.0):
    """train

    Parameters
    ----------
    txt : TYPE
        Description
    batch_size : int, optional
        Description
    sequence_length : int, optional
        Description
    n_cells : int, optional
        Description
    n_layers : int, optional
        Description
    learning_rate : float, optional
        Description
    max_iter : int, optional
        Description
    gradient_clip : float, optional
        Description
    ckpt_name : str, optional
        Description
    keep_prob : float, optional
        Description

    Returns
    -------
    TYPE
        Description
    """
    g = tf.Graph()
    with tf.Session(graph=g) as sess:
        model = build_model(
            txt=txt,
            batch_size=batch_size,
            sequence_length=sequence_length,
            n_layers=n_layers,
            n_cells=n_cells,
            gradient_clip=gradient_clip,
            learning_rate=learning_rate)

        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        saver = tf.train.Saver()
        sess.run(init_op)
        if os.path.exists(ckpt_name + '.index') or os.path.exists(ckpt_name):
            saver.restore(sess, ckpt_name)
            print("Model restored.")

        cursor = 0
        it_i = 0
        print_step = 1000
        avg_cost = 0
        while it_i < max_iter:
            Xs, Ys = [], []
            for batch_i in range(batch_size):
                Xs.append([
                    model['encoder'][ch]
                    for ch in txt[cursor:cursor + sequence_length]
                ])
                Ys.append([
                    model['encoder'][ch]
                    for ch in txt[cursor + 1:cursor + sequence_length + 1]
                ])
                cursor += sequence_length
                if (cursor + 1) >= len(txt) - sequence_length - 1:
                    cursor = np.random.randint(0, high=sequence_length)

            feed_dict = {
                model['X']: Xs,
                model['Y']: Ys,
                model['keep_prob']: keep_prob
            }
            out = sess.run(
                [model['cost'], model['updates']], feed_dict=feed_dict)
            avg_cost += out[0]

            if (it_i + 1) % print_step == 0:
                p = sess.run(
                    model['probs'],
                    feed_dict={
                        model['X']: np.array(Xs[-1])[np.newaxis],
                        model['keep_prob']: 1.0
                    })
                print(p.shape, 'min:',
                      np.min(p), 'max:',
                      np.max(p), 'mean:', np.mean(p), 'std:', np.std(p))
                if isinstance(txt[0], str):
                    # Print original string
                    print('original:',
                          "".join([model['decoder'][ch] for ch in Xs[-1]]))

                    # Print max guess
                    amax = []
                    for p_i in p:
                        amax.append(model['decoder'][np.argmax(p_i)])
                    print('synth(amax):', "".join(amax))

                    # Print w/ sampling
                    samp = []
                    for p_i in p:
                        p_i = p_i.astype(np.float64)
                        p_i = p_i / p_i.sum()
                        idx = np.argmax(np.random.multinomial(1, p_i.ravel()))
                        samp.append(model['decoder'][idx])
                    print('synth(samp):', "".join(samp))

                print(it_i, avg_cost / print_step)
                avg_cost = 0

                save_path = saver.save(sess, ckpt_name, global_step=it_i)
                print("Model saved in file: %s" % save_path)

            print(it_i, out[0], end='\r')
            it_i += 1

        return model


def infer(txt,
          ckpt_name,
          n_iterations,
          n_cells=200,
          n_layers=3,
          learning_rate=0.001,
          max_iter=5000,
          gradient_clip=10.0,
          init_value=[0],
          keep_prob=1.0,
          sampling='prob',
          temperature=1.0):
    """infer

    Parameters
    ----------
    txt : TYPE
        Description
    ckpt_name : TYPE
        Description
    n_iterations : TYPE
        Description
    n_cells : int, optional
        Description
    n_layers : int, optional
        Description
    learning_rate : float, optional
        Description
    max_iter : int, optional
        Description
    gradient_clip : float, optional
        Description
    init_value : list, optional
        Description
    keep_prob : float, optional
        Description
    sampling : str, optional
        Description
    temperature : float, optional
        Description

    Returns
    -------
    TYPE
        Description
    """
    g = tf.Graph()
    with tf.Session(graph=g) as sess:
        sequence_length = len(init_value)
        model = build_model(
            txt=txt,
            batch_size=1,
            sequence_length=sequence_length,
            n_layers=n_layers,
            n_cells=n_cells,
            gradient_clip=gradient_clip,
            learning_rate=learning_rate)

        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        saver = tf.train.Saver()
        sess.run(init_op)
        if os.path.exists(ckpt_name):
            saver.restore(sess, ckpt_name)
            print("Model restored.")

        state = []
        synth = [init_value]
        for s_i in model['final_state']:
            state += sess.run(
                [s_i.c, s_i.h],
                feed_dict={
                    model['X']: [synth[-1]],
                    model['keep_prob']: keep_prob
                })

        for i in range(n_iterations):
            # print('iteration: {}/{}'.format(i, n_iterations), end='\r')
            feed_dict = {model['X']: [synth[-1]], model['keep_prob']: keep_prob}
            state_updates = []
            for state_i in range(n_layers):
                feed_dict[model['initial_state'][state_i].c] = \
                    state[state_i * 2]
                feed_dict[model['initial_state'][state_i].h] = state[state_i * 2
                                                                     + 1]
                state_updates.append(model['final_state'][state_i].c)
                state_updates.append(model['final_state'][state_i].h)
            p = sess.run(model['probs'], feed_dict=feed_dict)[0]
            if sampling == 'max':
                p = np.argmax(p)
            else:
                p = p.astype(np.float64)
                p = np.log(p) / temperature
                p = np.exp(p) / np.sum(np.exp(p))
                p = np.random.multinomial(1, p.ravel())
                p = np.argmax(p)
            # Get the current state
            state = [
                sess.run(s_i, feed_dict=feed_dict) for s_i in state_updates
            ]
            synth.append([p])
            print(model['decoder'][p], end='')
            sys.stdout.flush()
            if model['decoder'][p] in ['.', '?', '!']:
                print('\n')
        print(np.concatenate(synth).shape)
    print("".join([model['decoder'][ch] for ch in np.concatenate(synth)]))
    return [model['decoder'][ch] for ch in np.concatenate(synth)]


def test_alice(max_iter=5):
    """Summary

    Parameters
    ----------
    max_iter : int, optional
        Description

    Returns
    -------
    TYPE
        Description
    """
    utils.download('https://s3.amazonaws.com/cadl/models/alice.txt.gz')
    with gzip.open('alice.txt.gz', 'rb') as fp:
        txt = fp.read().decode('utf-8')
    return train(txt, n_layers=2, n_cells=20, max_iter=max_iter)


def test_trump(max_iter=100):
    """Summary

    Parameters
    ----------
    max_iter : int, optional
        Description
    """
    utils.download(
        'https://s3.amazonaws.com/cadl/models/trump.ckpt.data-00000-of-00001')
    utils.download('https://s3.amazonaws.com/cadl/models/trump.ckpt.meta')
    utils.download('https://s3.amazonaws.com/cadl/models/trump.ckpt.index')
    utils.download('https://s3.amazonaws.com/cadl/models/trump.txt')
    with open('trump.txt', 'r') as fp:
        txt = fp.read()
    #train(txt, ckpt_name='trump', max_iter=max_iter)
    print(infer(txt, ckpt_name='./trump.ckpt', n_iterations=max_iter))


def test_wtc():
    """Summary
    """
    from scipy.io.wavfile import write, read
    rate, aud = read('wtc.wav')
    txt = np.int8(np.round(aud / 16384.0 * 128.0))
    txt = np.squeeze(txt).tolist()
    # try with more than 100 iterations, e.g. 50k - 200k
    train(txt, sequence_length=250, n_layers=3, n_cells=512, max_iter=100)
    synthesis = infer(
        txt,
        './model.ckpt',
        8000 * 30,
        n_layers=3,
        n_cells=150,
        keep_prob=1.0,
        sampling='prob')
    snd = np.int16(np.array(synthesis) / 128.0 * 16384.0)
    write('wtc-synth.wav', 8000, snd)


if __name__ == '__main__':
    test_alice()
