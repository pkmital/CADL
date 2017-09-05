"""Sequence to Sequence models w/ Attention and BiDirectional Dynamic RNNs.

Parag K. Mital
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


def _create_encoder(embed, lengths, batch_size, n_enc_neurons, n_layers,
                    use_lstm):
    # Create the RNN Cells for encoder
    if use_lstm:
        cell_fw = tf.contrib.rnn.BasicLSTMCell(n_enc_neurons)
    else:
        cell_fw = tf.contrib.rnn.GRUCell(n_enc_neurons)

    # Build deeper recurrent net if using more than 1 layer
    if n_layers > 1:
        cell_fw = tf.contrib.rnn.MultiRNNCell([cell_fw] * n_layers)

    # Create the internal multi-layer cell for the backward RNN.
    if use_lstm:
        cell_bw = tf.contrib.rnn.BasicLSTMCell(n_enc_neurons)
    else:
        cell_bw = tf.contrib.rnn.GRUCell(n_enc_neurons)

    # Build deeper recurrent net if using more than 1 layer
    if n_layers > 1:
        cell_bw = tf.contrib.rnn.MultiRNNCell([cell_bw] * n_layers)

    # Now hookup the cells to the input
    # [batch_size, max_time, embed_size]
    # We only use the forward cell's final state since the decoder is
    # not a bidirectional rnn
    (_, final_state) = tf.nn.bidirectional_dynamic_rnn(
        cell_fw=cell_fw,
        cell_bw=cell_bw,
        inputs=embed,
        sequence_length=lengths,
        time_major=False,
        dtype=tf.float32)
    return final_state


def _create_train_decoder(cells, encoder_state, encoding_lengths, decoding,
                          decoding_lengths, embed_matrix, batch_size,
                          target_vocab_size, use_attention, n_dec_neurons,
                          scope, output_fn, max_sequence_size):

    if use_attention:
        attention_states = tf.zeros([batch_size, 1, cells.output_size])
        # Pass in the final hidden states of the encoder's RNN which it will
        # attend over... thus determining which ones are useful for the
        # decoding.
        (attn_keys, attn_vals, attn_score_fn, attn_construct_fn) = \
            tf.contrib.seq2seq.prepare_attention(
                attention_states=attention_states,
                attention_option='bahdanau',
                num_units=n_dec_neurons)

        # Use the final state of the encoder as input and build a decoder also
        # taking information from the attention module acting on the encoder_state.
        decoder_fn = \
            tf.contrib.seq2seq.attention_decoder_fn_train(
                encoder_state=encoder_state,
                attention_keys=attn_keys,
                attention_values=attn_vals,
                attention_score_fn=attn_score_fn,
                attention_construct_fn=attn_construct_fn)

    else:
        # Build training decoder function
        decoder_fn = \
            tf.contrib.seq2seq.simple_decoder_fn_train(
                encoder_state=encoder_state)

    # Build training rnn decoder
    outputs, _, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(
        cell=cells,
        decoder_fn=decoder_fn,
        inputs=decoding,
        sequence_length=decoding_lengths,
        time_major=False,
        scope=scope)

    # Convert to vocab size
    train_logits = output_fn(outputs)

    return train_logits


def _create_inference_decoder(cells, encoder_state, encoding_lengths, decoding,
                              decoding_lengths, embed_matrix, batch_size,
                              n_dec_neurons, target_vocab_size, use_attention,
                              scope, output_fn, max_sequence_size):

    if use_attention:
        attention_states = tf.zeros([batch_size, 1, cells.output_size])
        # Pass in the final hidden states of the encoder's RNN which it will
        # attend over... thus determining which ones are useful for the
        # decoding.
        (attn_keys, attn_vals, attn_score_fn, attn_construct_fn) = \
            tf.contrib.seq2seq.prepare_attention(
                attention_states=attention_states,
                attention_option='bahdanau',
                num_units=n_dec_neurons)

        # Build a separate inference network to use during generation.
        decoder_fn_inference = \
            tf.contrib.seq2seq.attention_decoder_fn_inference(
                output_fn=output_fn,
                encoder_state=encoder_state,
                attention_keys=attn_keys,
                attention_values=attn_vals,
                attention_score_fn=attn_score_fn,
                attention_construct_fn=attn_construct_fn,
                embeddings=embed_matrix,
                start_of_sequence_id=GO_ID,
                end_of_sequence_id=EOS_ID,
                maximum_length=max_sequence_size,
                num_decoder_symbols=target_vocab_size)
    else:
        # Build inference decoder function
        decoder_fn_inference = \
            tf.contrib.seq2seq.simple_decoder_fn_inference(
                output_fn=output_fn,
                encoder_state=encoder_state,
                embeddings=embed_matrix,
                start_of_sequence_id=GO_ID,
                end_of_sequence_id=EOS_ID,
                maximum_length=max_sequence_size,
                num_decoder_symbols=target_vocab_size)

    # Build inference rnn decoder (handles output to vocab size, so we
    # do not have to apply the output function).
    (infer_logits, _, _) = tf.contrib.seq2seq.dynamic_rnn_decoder(
        cell=cells,
        decoder_fn=decoder_fn_inference,
        time_major=False,
        scope=scope)

    return infer_logits


def create_model(source_vocab_size=20000,
                 target_vocab_size=20000,
                 input_embed_size=1024,
                 target_embed_size=1024,
                 share_input_and_target_embedding=True,
                 n_neurons=512,
                 n_layers=3,
                 use_lstm=True,
                 use_attention=True,
                 max_sequence_size=50):

    n_enc_neurons = n_neurons
    n_dec_neurons = n_neurons

    # First sentence (i.e. input, original language sentence before translation)
    # [batch_size, max_time]
    source = tf.placeholder(tf.int32, shape=(None, None), name='source')

    # User should also pass in the sequence lengths
    source_lengths = tf.placeholder(
        tf.int32, shape=(None), name='source_lengths')

    # Second sentence (i.e. reply, translation, etc...)
    # [batch_size, max_time]
    target = tf.placeholder(tf.int32, shape=(None, None), name='target')

    # User should also pass in the sequence lengths
    target_lengths = tf.placeholder(
        tf.int32, shape=(None), name='target_lengths')

    # Get symbolic shapes
    batch_size, sequence_size = tf.unstack(tf.shape(source))

    with tf.variable_scope('target/slicing'):
        slice = tf.slice(target, [0, 0], [batch_size, -1])
        decoder_input = tf.concat([tf.fill([batch_size, 1], GO_ID), slice], 1)

    with tf.variable_scope('source/embedding'):
        source_embed, source_embed_matrix = _create_embedding(
            x=source, vocab_size=source_vocab_size, embed_size=input_embed_size)

    with tf.variable_scope('target/embedding'):
        # Check if we need a new embedding matrix or not.  If we are for
        # instance translating to another language, then we'd need different
        # vocabularies for the input and outputs, and so new embeddings.
        # However if we are for instance building a chatbot with the same
        # language, then it doesn't make sense to have different embeddings and
        # we should share them.
        if (share_input_and_target_embedding and
                source_vocab_size == target_vocab_size):
            target_embed, target_embed_matrix = _create_embedding(
                x=decoder_input,
                vocab_size=target_vocab_size,
                embed_size=target_embed_size,
                embed_matrix=source_embed_matrix)
        elif source_vocab_size != target_vocab_size:
            raise ValueError(
                'source_vocab_size must equal target_vocab_size if ' +
                'sharing input and target embeddings')
        else:
            target_embed, target_embed_matrix = _create_embedding(
                x=target,
                vocab_size=target_vocab_size,
                embed_size=target_embed_size)

    # Build the encoder
    with tf.variable_scope('encoder'):
        encoder_state = _create_encoder(
            embed=source_embed,
            lengths=source_lengths,
            batch_size=batch_size,
            n_enc_neurons=n_enc_neurons,
            n_layers=n_layers,
            use_lstm=use_lstm)

    # Build the decoder
    with tf.variable_scope('decoder') as scope:

        def output_fn(x):
            return tf.contrib.layers.fully_connected(
                inputs=x,
                num_outputs=target_vocab_size,
                activation_fn=None,
                scope=scope)

        # Create the RNN Cells for decoder
        if use_lstm:
            cells = tf.contrib.rnn.BasicLSTMCell(n_dec_neurons)
        else:
            cells = tf.contrib.rnn.GRUCell(n_dec_neurons)

        # Build deeper recurrent net if using more than 1 layer
        if n_layers > 1:
            cells = tf.contrib.rnn.MultiRNNCell([cells] * n_layers)

        decoding_train = _create_train_decoder(
            cells=cells,
            encoder_state=encoder_state[0],
            encoding_lengths=source_lengths,
            decoding=target_embed,
            decoding_lengths=target_lengths,
            embed_matrix=target_embed_matrix,
            batch_size=batch_size,
            target_vocab_size=target_vocab_size,
            use_attention=use_attention,
            scope=scope,
            max_sequence_size=max_sequence_size,
            n_dec_neurons=n_dec_neurons,
            output_fn=output_fn)

        # Inference model:
        scope.reuse_variables()
        decoding_inference = _create_inference_decoder(
            cells=cells,
            encoder_state=encoder_state[0],
            encoding_lengths=source_lengths,
            decoding=target_embed,
            decoding_lengths=target_lengths,
            embed_matrix=target_embed_matrix,
            batch_size=batch_size,
            target_vocab_size=target_vocab_size,
            use_attention=use_attention,
            scope=scope,
            max_sequence_size=max_sequence_size,
            n_dec_neurons=n_dec_neurons,
            output_fn=output_fn)

    with tf.variable_scope('loss'):
        weights = tf.ones(
            [batch_size, tf.reduce_max(target_lengths)],
            dtype=tf.float32,
            name="weights")
        loss = tf.contrib.seq2seq.sequence_loss(
            logits=tf.reshape(decoding_train, [
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
        'thought_vector': encoder_state,
        'decoder': decoding_inference
    }


def batch_generator(Xs, Ys, source_lengths, target_lengths, batch_size=50):
    idxs = np.random.permutation(np.arange(len(Xs)))
    n_batches = len(idxs) // batch_size
    for batch_i in range(n_batches):
        this_idxs = idxs[batch_i * batch_size:(batch_i + 1) * batch_size]
        this_Xs, this_Ys = Xs[this_idxs, :], Ys[this_idxs, :]
        this_source_lengths, this_target_lengths = source_lengths[
            this_idxs], target_lengths[this_idxs]
        yield (this_Xs[:, :np.max(this_source_lengths)],
               this_Ys[:, :np.max(this_target_lengths)], this_source_lengths,
               this_target_lengths)


def preprocess(text, min_count=10, max_length=50):
    sentences = [el for s in text for el in nltk.sent_tokenize(s)]

    # We'll first tokenize each sentence into words to get a sense of
    # how long each sentence is:
    words = [[word.lower() for word in nltk.word_tokenize(s)]
             for s in sentences]

    # Then see how long each sentence is:
    lengths = np.array([len(s) for s in words])

    good_idxs = np.where(lengths <= max_length)[0]
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
    unked = []
    for s in words:
        this_sentence = [vocab.get(w, UNK_ID) for w in s]
        unked.append(this_sentence)
    return unked


def id2word(ids, vocab):
    words = []
    id2words = {v: k for k, v in vocab.items()}
    for s in ids:
        this_sentence = [id2words.get(w) for w in s]
        words.append(this_sentence)
    return words


def test_cornell():

    # Get the cornell dataset text
    text = cornell.get_scripts()

    # Preprocess it to word IDs including UNKs for out of vocabulary words
    max_sequence_size = 50
    unked, vocab = preprocess(
        text, min_count=10, max_length=max_sequence_size - 1)

    # Get the vocabulary size
    vocab_size = len(vocab)

    # Create input output pairs formed by neighboring sentences of dialog
    Xs_list, Ys_list = unked[:-1], unked[1:]

    # Store the final lengths
    source_lengths = np.zeros((len(Xs_list)), dtype=np.int32)
    target_lengths = np.zeros((len(Ys_list)), dtype=np.int32)
    Xs = np.ones((len(Xs_list), max_sequence_size), dtype=np.int32) * PAD_ID
    Ys = np.ones((len(Ys_list), max_sequence_size), dtype=np.int32) * PAD_ID

    for i, (source_i, target_i) in enumerate(zip(Xs_list, Ys_list)):
        el = source_i
        source_lengths[i] = len(el)
        Xs[i, :len(el)] = el

        el = target_i + [EOS_ID]
        target_lengths[i] = len(el)
        Ys[i, :len(el)] = el

    sess = tf.Session()

    net = create_model(
        use_attention=True,
        source_vocab_size=vocab_size,
        target_vocab_size=vocab_size)
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')
    current_learning_rate = 0.01
    opt = tf.train.AdamOptimizer(
        learning_rate=learning_rate).minimize(net['loss'])
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    sess.run(init_op)
    saver = tf.train.Saver()

    def decode(sentence):
        preprocessed = [
            word
            for s in nltk.sent_tokenize(sentence.lower())
            for word in nltk.word_tokenize(s)
        ][::-1]
        tokens = cornell.word2id([preprocessed + [_GO]], vocab)
        outputs = sess.run(
            net['decoder'],
            feed_dict={
                net['source']: tokens,
                net['source_lengths']: [len(x_i) for x_i in tokens]
            })
        decoding = np.argmax(outputs, axis=2)
        print('input:', sentence, '\n', 'output:',
              " ".join(cornell.id2word(decoding, vocab)[0]))

    n_epochs = 10
    batch_size = 50
    for epoch_i in range(n_epochs):
        for it_i, (this_Xs, this_Ys, this_source_lengths, this_target_lengths) \
                    in enumerate(batch_generator(
                        Xs, Ys, source_lengths, target_lengths, batch_size=batch_size)):
            if it_i % 100 == 0:
                current_learning_rate = current_learning_rate * 0.9
                rand_idx = np.random.randint(0, high=len(text))
                print(it_i)
                decode(text[rand_idx])
            l = sess.run(
                [net['loss'], opt],
                feed_dict={
                    learning_rate: current_learning_rate,
                    net['source']: this_Xs,
                    net['target']: this_Ys,
                    net['source_lengths']: this_source_lengths,
                    net['target_lengths']: this_target_lengths
                })[0]
            print('{}: {}'.format(it_i, l), end='\r')
        # End of epoch, save
        saver.save(sess, './dynamic-seq2seq.ckpt', global_step=it_i)

    sess.close()
