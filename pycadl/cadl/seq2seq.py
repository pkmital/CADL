"""Sequence to Sequence models w/ Attention and BiDirectional Dynamic RNNs.
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
import nltk
import pickle
from cadl import cornell

# Special vocabulary symbols:
# PAD is used to pad a sequence to a fixed size
# GO is for the end of the encoding
# EOS is for the end of decoding
# UNK is for out of vocabulary words
_PAD, _GO, _EOS, _UNK = "_PAD", "_GO", "_EOS", "_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]
PAD_ID, GO_ID, EOS_ID, UNK_ID = range(4)


def _create_embedding(x, vocab_size, embed_size, embed_matrix=None):
    """Summary

    Parameters
    ----------
    x : TYPE
        Description
    vocab_size : TYPE
        Description
    embed_size : TYPE
        Description
    embed_matrix : None, optional
        Description

    Returns
    -------
    TYPE
        Description
    """
    # Creating an embedding matrix if one isn't given
    if embed_matrix is None:
        # This is a big matrix
        embed_matrix = tf.get_variable(
            name="embedding_matrix",
            shape=[vocab_size, embed_size],
            dtype=tf.float32,
            initializer=tf.random_uniform_initializer(-1.0, 1.0))

    # Perform the lookup of ids in x and perform the embedding to embed_size
    # [batch_size, max_time, embed_size]
    embed = tf.nn.embedding_lookup(embed_matrix, x)

    return embed, embed_matrix


def _create_rnn_cell(n_neurons, n_layers, keep_prob):
    """Summary

    Parameters
    ----------
    n_neurons : TYPE
        Description
    n_layers : TYPE
        Description
    keep_prob : TYPE
        Description

    Returns
    -------
    TYPE
        Description
    """
    import tensorflow.contrib.rnn as rnn

    cell_fw = rnn.LayerNormBasicLSTMCell(
        num_units=n_neurons, dropout_keep_prob=keep_prob)
    # Build deeper recurrent net if using more than 1 layer
    if n_layers > 1:
        cells = [cell_fw]
        for layer_i in range(1, n_layers):
            with tf.variable_scope('{}'.format(layer_i)):
                cell_fw = rnn.LayerNormBasicLSTMCell(
                    num_units=n_neurons, dropout_keep_prob=keep_prob)
                cells.append(cell_fw)
        cell_fw = rnn.MultiRNNCell(cells)
    return cell_fw


def _create_encoder(embed, lengths, batch_size, n_enc_neurons, n_layers,
                    keep_prob):
    """Summary

    Parameters
    ----------
    embed : TYPE
        Description
    lengths : TYPE
        Description
    batch_size : TYPE
        Description
    n_enc_neurons : TYPE
        Description
    n_layers : TYPE
        Description
    keep_prob : TYPE
        Description

    Returns
    -------
    TYPE
        Description
    """
    # Create the RNN Cells for encoder
    with tf.variable_scope('forward'):
        cell_fw = _create_rnn_cell(n_enc_neurons, n_layers, keep_prob)

    # Create the internal multi-layer cell for the backward RNN.
    with tf.variable_scope('backward'):
        cell_bw = _create_rnn_cell(n_enc_neurons, n_layers, keep_prob)

    # Now hookup the cells to the input
    # [batch_size, max_time, embed_size]
    (outputs, final_state) = tf.nn.bidirectional_dynamic_rnn(
        cell_fw=cell_fw,
        cell_bw=cell_bw,
        inputs=embed,
        sequence_length=lengths,
        time_major=False,
        dtype=tf.float32)

    return outputs, final_state


def _create_decoder(cells,
                    batch_size,
                    encoder_outputs,
                    encoder_state,
                    encoder_lengths,
                    decoding_inputs,
                    decoding_lengths,
                    embed_matrix,
                    target_vocab_size,
                    scope,
                    max_sequence_size,
                    use_attention=True):
    """Summary

    Parameters
    ----------
    cells : TYPE
        Description
    batch_size : TYPE
        Description
    encoder_outputs : TYPE
        Description
    encoder_state : TYPE
        Description
    encoder_lengths : TYPE
        Description
    decoding_inputs : TYPE
        Description
    decoding_lengths : TYPE
        Description
    embed_matrix : TYPE
        Description
    target_vocab_size : TYPE
        Description
    scope : TYPE
        Description
    max_sequence_size : TYPE
        Description
    use_attention : bool, optional
        Description

    Returns
    -------
    TYPE
        Description
    """
    from tensorflow.python.layers.core import Dense

    # Output projection
    output_layer = Dense(target_vocab_size, name='output_projection')

    # Setup Attention
    if use_attention:
        attn_mech = tf.contrib.seq2seq.LuongAttention(
            cells.output_size, encoder_outputs, encoder_lengths, scale=True)
        cells = tf.contrib.seq2seq.AttentionWrapper(
            cell=cells,
            attention_mechanism=attn_mech,
            attention_layer_size=cells.output_size,
            alignment_history=False)
        initial_state = cells.zero_state(
            dtype=tf.float32, batch_size=batch_size)
        initial_state = initial_state.clone(cell_state=encoder_state)

    # Setup training a build decoder
    helper = tf.contrib.seq2seq.TrainingHelper(
        inputs=decoding_inputs,
        sequence_length=decoding_lengths,
        time_major=False)
    train_decoder = tf.contrib.seq2seq.BasicDecoder(
        cell=cells,
        helper=helper,
        initial_state=initial_state,
        output_layer=output_layer)
    train_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
        train_decoder,
        output_time_major=False,
        impute_finished=True,
        maximum_iterations=max_sequence_size)
    train_logits = tf.identity(train_outputs.rnn_output, name='train_logits')

    # Setup inference and build decoder
    scope.reuse_variables()
    start_tokens = tf.tile(tf.constant([GO_ID], dtype=tf.int32), [batch_size])
    helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
        embedding=embed_matrix, start_tokens=start_tokens, end_token=EOS_ID)
    infer_decoder = tf.contrib.seq2seq.BasicDecoder(
        cell=cells,
        helper=helper,
        initial_state=initial_state,
        output_layer=output_layer)
    infer_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
        infer_decoder,
        output_time_major=False,
        impute_finished=True,
        maximum_iterations=max_sequence_size)
    infer_logits = tf.identity(infer_outputs.sample_id, name='infer_logits')

    return train_logits, infer_logits


def create_model(source_vocab_size=10000,
                 target_vocab_size=10000,
                 input_embed_size=512,
                 target_embed_size=512,
                 share_input_and_target_embedding=True,
                 n_neurons=512,
                 n_layers=4,
                 use_attention=True,
                 max_sequence_size=30):
    """Summary

    Parameters
    ----------
    source_vocab_size : int, optional
        Description
    target_vocab_size : int, optional
        Description
    input_embed_size : int, optional
        Description
    target_embed_size : int, optional
        Description
    share_input_and_target_embedding : bool, optional
        Description
    n_neurons : int, optional
        Description
    n_layers : int, optional
        Description
    use_attention : bool, optional
        Description
    max_sequence_size : int, optional
        Description

    Returns
    -------
    TYPE
        Description

    Raises
    ------
    ValueError
        Description
    """
    n_enc_neurons = n_neurons
    n_dec_neurons = n_neurons

    # First sentence (i.e. input, original language sentence before translation)
    # [batch_size, max_time]
    source = tf.placeholder(tf.int32, shape=(None, None), name='source')

    # User should also pass in the sequence lengths
    source_lengths = tf.placeholder(
        tf.int32, shape=(None,), name='source_lengths')

    # Second sentence (i.e. reply, translation, etc...)
    # [batch_size, max_time]
    target = tf.placeholder(tf.int32, shape=(None, None), name='target')

    # User should also pass in the sequence lengths
    target_lengths = tf.placeholder(
        tf.int32, shape=(None,), name='target_lengths')

    # Dropout
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    # Symbolic shapes
    batch_size, sequence_length = tf.unstack(tf.shape(source))

    # Get the input to the decoder by removing last element
    # and adding a 'go' symbol as first element
    with tf.variable_scope('target/slicing'):
        slice = tf.slice(target, [0, 0], [batch_size, -1])
        decoder_input = tf.concat([tf.fill([batch_size, 1], GO_ID), slice], 1)

    # Embed word ids to target embedding
    with tf.variable_scope('source/embedding'):
        source_embed, source_embed_matrix = _create_embedding(
            x=source, vocab_size=source_vocab_size, embed_size=input_embed_size)

    # Embed word ids for target embedding
    with tf.variable_scope('target/embedding'):
        # Check if we need a new embedding matrix or not.  If we are for
        # instance translating to another language, then we'd need different
        # vocabularies for the input and outputs, and so new embeddings.
        # However if we are for instance building a chatbot with the same
        # language, then it doesn't make sense to have different embeddings and
        # we should share them.
        if (share_input_and_target_embedding and
                source_vocab_size == target_vocab_size):
            target_input_embed, target_embed_matrix = _create_embedding(
                x=decoder_input,
                vocab_size=target_vocab_size,
                embed_size=target_embed_size,
                embed_matrix=source_embed_matrix)
        elif source_vocab_size != target_vocab_size:
            raise ValueError(
                'source_vocab_size must equal target_vocab_size if ' +
                'sharing input and target embeddings')
        else:
            target_input_embed, target_embed_matrix = _create_embedding(
                x=target,
                vocab_size=target_vocab_size,
                embed_size=target_embed_size)

    # Build the encoder
    with tf.variable_scope('encoder'):
        encoder_outputs, encoder_state = _create_encoder(
            embed=source_embed,
            lengths=source_lengths,
            batch_size=batch_size,
            n_enc_neurons=n_enc_neurons,
            n_layers=n_layers,
            keep_prob=keep_prob)

    # Build the decoder
    with tf.variable_scope('decoder') as scope:
        cell_fw = _create_rnn_cell(n_dec_neurons, n_layers, keep_prob)
        decoding_train_logits, decoding_infer_logits = _create_decoder(
            cells=cell_fw,
            batch_size=batch_size,
            encoder_outputs=encoder_outputs[0],
            encoder_state=encoder_state[0],
            encoder_lengths=source_lengths,
            decoding_inputs=target_input_embed,
            decoding_lengths=target_lengths,
            embed_matrix=target_embed_matrix,
            target_vocab_size=target_vocab_size,
            scope=scope,
            max_sequence_size=max_sequence_size)

    with tf.variable_scope('loss'):
        weights = tf.cast(tf.sequence_mask(target_lengths), tf.float32)
        loss = tf.contrib.seq2seq.sequence_loss(
            logits=tf.reshape(decoding_train_logits, [
                batch_size, tf.reduce_max(target_lengths), target_vocab_size
            ]),
            targets=target,
            weights=weights)

    return {
        'loss': loss,
        'source': source,
        'source_lengths': source_lengths,
        'target': target,
        'target_lengths': target_lengths,
        'keep_prob': keep_prob,
        'thought_vector': encoder_state,
        'decoder': decoding_infer_logits
    }


def batch_generator(sources,
                    targets,
                    source_lengths,
                    target_lengths,
                    batch_size=50):
    """Summary

    Parameters
    ----------
    sources : TYPE
        Description
    targets : TYPE
        Description
    source_lengths : TYPE
        Description
    target_lengths : TYPE
        Description
    batch_size : int, optional
        Description

    Yields
    ------
    TYPE
        Description
    """
    idxs = np.random.permutation(np.arange(len(sources)))
    n_batches = len(idxs) // batch_size
    for batch_i in range(n_batches):
        this_idxs = idxs[batch_i * batch_size:(batch_i + 1) * batch_size]
        this_sources, this_targets = sources[this_idxs, :], targets[
            this_idxs, :]
        this_source_lengths, this_target_lengths = source_lengths[
            this_idxs], target_lengths[this_idxs]
        yield (this_sources[:, :np.max(this_source_lengths)],
               this_targets[:, :np.max(this_target_lengths)],
               this_source_lengths, this_target_lengths)


def preprocess(text, min_count=5, min_length=3, max_length=30):
    """Summary

    Parameters
    ----------
    text : TYPE
        Description
    min_count : int, optional
        Description
    min_length : int, optional
        Description
    max_length : int, optional
        Description

    Returns
    -------
    TYPE
        Description
    """
    sentences = [el for s in text for el in nltk.sent_tokenize(s)]

    # We'll first tokenize each sentence into words to get a sense of
    # how long each sentence is:
    words = [[word.lower() for word in nltk.word_tokenize(s)]
             for s in sentences]

    # Then see how long each sentence is:
    lengths = np.array([len(s) for s in words])

    good_idxs = np.where((lengths >= min_length) & (lengths < max_length))[0]
    dataset = [words[idx] for idx in good_idxs]
    fdist = nltk.FreqDist([word for sentence in dataset for word in sentence])

    vocab_counts = [el for el in fdist.most_common() if el[1] > min_count]

    # First sort the vocabulary
    vocab = [v[0] for v in vocab_counts]
    vocab.sort()

    # Now add the special symbols:
    vocab = _START_VOCAB + vocab

    # Then create the word to id mapping
    vocab = {k: v for v, k in enumerate(vocab)}

    with open('vocab.pkl', 'wb') as fp:
        pickle.dump(vocab, fp)

    unked = word2id(dataset, vocab)
    return unked, vocab


def word2id(words, vocab):
    """Summary

    Parameters
    ----------
    words : TYPE
        Description
    vocab : TYPE
        Description

    Returns
    -------
    TYPE
        Description
    """
    unked = []
    for s in words:
        this_sentence = [vocab.get(w, UNK_ID) for w in s]
        unked.append(this_sentence)
    return unked


def id2word(ids, vocab):
    """Summary

    Parameters
    ----------
    ids : TYPE
        Description
    vocab : TYPE
        Description

    Returns
    -------
    TYPE
        Description
    """
    words = []
    id2words = {v: k for k, v in vocab.items()}
    for s in ids:
        this_sentence = [id2words.get(w) for w in s]
        words.append(this_sentence)
    return words


def train(text,
          max_sequence_size=20,
          use_attention=True,
          min_count=25,
          min_length=5,
          n_epochs=1000,
          batch_size=100):
    """Summary

    Parameters
    ----------
    text : TYPE
        Description
    max_sequence_size : int, optional
        Description
    use_attention : bool, optional
        Description
    min_count : int, optional
        Description
    min_length : int, optional
        Description
    n_epochs : int, optional
        Description
    batch_size : int, optional
        Description
    """
    # Preprocess it to word IDs including UNKs for out of vocabulary words
    unked, vocab = preprocess(
        text,
        min_count=min_count,
        min_length=min_length,
        max_length=max_sequence_size - 1)

    # Get the vocabulary size
    vocab_size = len(vocab)

    # Create input output pairs formed by neighboring sentences of dialog
    sources_list, targets_list = unked[:-1], unked[1:]

    # Store the final lengths
    source_lengths = np.zeros((len(sources_list)), dtype=np.int32)
    target_lengths = np.zeros((len(targets_list)), dtype=np.int32)
    sources = np.ones(
        (len(sources_list), max_sequence_size), dtype=np.int32) * PAD_ID
    targets = np.ones(
        (len(targets_list), max_sequence_size), dtype=np.int32) * PAD_ID

    for i, (source_i, target_i) in enumerate(zip(sources_list, targets_list)):
        el = source_i
        source_lengths[i] = len(el)
        sources[i, :len(el)] = el

        el = target_i + [EOS_ID]
        target_lengths[i] = len(el)
        targets[i, :len(el)] = el

    sess = tf.Session()

    net = create_model(
        max_sequence_size=max_sequence_size,
        use_attention=use_attention,
        source_vocab_size=vocab_size,
        target_vocab_size=vocab_size)

    learning_rate = tf.placeholder(tf.float32, name='learning_rate')
    opt = tf.train.AdamOptimizer(
        learning_rate=learning_rate).minimize(net['loss'])
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    sess.run(init_op)
    saver = tf.train.Saver()

    def decode(tokens, lengths):
        """Summary

        Parameters
        ----------
        tokens : TYPE
            Description
        lengths : TYPE
            Description
        """
        decoding = sess.run(
            net['decoder'],
            feed_dict={
                net['keep_prob']: 1.0,
                net['source']: tokens,
                net['source_lengths']: lengths
            })
        print('input:', " ".join(id2word([tokens[0]], vocab)[0]))
        print('output:', " ".join(id2word([decoding[0]], vocab)[0]))

    current_learning_rate = 0.01
    for epoch_i in range(n_epochs):
        total = 0
        for it_i, (this_sources, this_targets, this_source_lengths, this_target_lengths) \
            in enumerate(batch_generator(
                sources, targets, source_lengths, target_lengths, batch_size=batch_size)):
            if it_i % 1000 == 0:
                current_learning_rate = max(0.0001,
                                            current_learning_rate * 0.99)
                print(it_i)
                decode(this_sources[0:1], this_source_lengths[0:1])
            l = sess.run(
                [net['loss'], opt],
                feed_dict={
                    learning_rate: current_learning_rate,
                    net['keep_prob']: 0.8,
                    net['source']: this_sources,
                    net['target']: this_targets,
                    net['source_lengths']: this_source_lengths,
                    net['target_lengths']: this_target_lengths
                })[0]
            total = total + l
            print('{}: {}'.format(it_i, total / (it_i + 1)), end='\r')
        # End of epoch, save
        print('epoch {}: {}'.format(epoch_i, total / it_i))
        saver.save(sess, './dynamic-seq2seq.ckpt', global_step=it_i)

    sess.close()


def train_cornell(**kwargs):
    """Summary

    Parameters
    ----------
    **kwargs
        Description

    Returns
    -------
    TYPE
        Description
    """
    # Get the cornell dataset text
    text = cornell.get_scripts()
    return train(text, **kwargs)
