"""Word2Vec model.
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


def build_model(batch_size=128, vocab_size=50000, embedding_size=128,
                n_neg_samples=64):
    """Summary

    Parameters
    ----------
    batch_size : int, optional
        Description
    vocab_size : int, optional
        Description
    embedding_size : int, optional
        Description
    n_neg_samples : int, optional
        Description

    Returns
    -------
    TYPE
        Description
    """
    # Input placeholders
    center_words = tf.placeholder(
        tf.int32, shape=[batch_size], name='center_words')
    target_words = tf.placeholder(
        tf.int32, shape=[batch_size, 1], name='target_words')

    # This is the important part of the model which will embed a word id
    # into an embedding of size `embedding_size`
    embed_matrix = tf.get_variable(
        name='embedding',
        shape=[vocab_size, embedding_size],
        dtype=tf.float32,
        initializer=tf.random_uniform_initializer(-1.0, 1.0))

    # Define the inference
    embed = tf.nn.embedding_lookup(embed_matrix, center_words, name='embed')

    # Construct variables for NCE loss
    nce_weight = tf.get_variable(
        name='nce/weight',
        shape=[vocab_size, embedding_size],
        dtype=tf.float32,
        initializer=tf.truncated_normal_initializer(
            stddev=1.0 / (embedding_size ** 0.5)))
    nce_bias = tf.get_variable(
        name='nce/bias',
        shape=[vocab_size],
        dtype=tf.float32,
        initializer=tf.constant_initializer())

    # Define loss function to be NCE loss function
    loss = tf.reduce_mean(tf.nn.nce_loss(
        weights=nce_weight,
        biases=nce_bias,
        labels=target_words,
        inputs=embed,
        num_sampled=n_neg_samples,
        num_classes=vocab_size), name='loss')

    return locals()
