"""Tools for downloading and preprocessing the Cornell Movie DB.
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

Attributes
----------
conversations : str
    Description
lines : str
    Description
titles : str
    Description
"""
import os
import ast
import nltk
import numpy as np
import pickle
import string
import bs4
import tensorflow as tf
from cadl import utils

titles = 'movie_titles_metadata.txt'
conversations = 'movie_conversations.txt'
lines = 'movie_lines.txt'


def download_cornell(dst='cornell movie-dialogs corpus'):
    """Summary

    Parameters
    ----------
    dst : str, optional
        Description
    """
    utils.download_and_extract_zip(
        'https://s3.amazonaws.com/cadl/models/cornell_movie_dialogs_corpus.zip',
        dst)


def get_characters(path='cornell movie-dialogs corpus'):
    '''
    - movie_characters_metadata.txt
        - contains information about each movie character
        - fields:
            - characterID
            - character name
            - movieID
            - movie title
            - gender ("?" for unlabeled cases)
            - position in credits ("?" for unlabeled cases)

    Parameters
    ----------
    path : TYPE, optional
        Description

    Returns
    -------
    TYPE
        Description
    '''
    characters = {}
    download_cornell(path)
    with open(
            os.path.join(path, 'movie_characters_metadata.txt'),
            'r',
            encoding='latin-1') as f:
        for line_i in f:
            els = [el.strip() for el in line_i.split('+++$+++')]
            characters[els[0]] = {
                'character_id': els[0],
                'name': els[1],
                'movie_id': els[2],
                'movie_name': els[3]
            }
    return characters


def get_titles(path='cornell movie-dialogs corpus'):
    '''
    - movie_titles_metadata.txt
        - contains information about each movie title
        - fields:
            - movieID,
            - movie title,
            - movie year,
            - IMDB rating,
            - no. IMDB votes,
            - genres in the format ['genre1','genre2',É,'genreN']

    Parameters
    ----------
    path : TYPE, optional
        Description

    Returns
    -------
    TYPE
        Description
    '''
    titles = {}
    download_cornell(path)
    with open(
            os.path.join(path, 'movie_titles_metadata.txt'),
            'r',
            encoding='latin-1') as f:
        for line_i in f:
            els = [el.strip() for el in line_i.split('+++$+++')]
            titles[els[0]] = {
                'movie_id': els[0],
                'name': els[1],
                'year': els[2],
                'imdb_rating': els[3],
                'imdb_num_votes': els[4],
                'genres': els[5]
            }
    return titles


def get_conversations(path='cornell movie-dialogs corpus'):
    '''
    - movie_conversations.txt
        - the structure of the conversations
        - fields
            - characterID of the first character involved in the conversation
            - characterID of the second character involved in the conversation
            - movieID of the movie in which the conversation occurred
            - list of the utterances that make the conversation, in
                chronological order: ['lineID1','lineID2',É,'lineIDN']
                has to be matched with movie_lines.txt to reconstruct the
                actual content

    Parameters
    ----------
    path : TYPE, optional
        Description

    Returns
    -------
    TYPE
        Description
    '''
    conversations = []
    download_cornell(path)
    with open(
            os.path.join(path, 'movie_conversations.txt'), 'r',
            encoding='latin-1') as f:
        for line_i in f:
            els = [el.strip() for el in line_i.split('+++$+++')]
            conversations.append({
                'character_id_1': els[0],
                'character_id_2': els[1],
                'movie_id': els[2],
                'lines': els[3]
            })
    return conversations


def get_lines(path='cornell movie-dialogs corpus'):
    '''
    - movie_lines.txt
        - contains the actual text of each utterance
        - fields:
            - lineID
            - characterID (who uttered this phrase)
            - movieID
            - character name
            - text of the utterance

    Parameters
    ----------
    path : TYPE, optional
        Description

    Returns
    -------
    TYPE
        Description
    '''
    lines = {}
    download_cornell(path)
    with open(
            os.path.join(path, 'movie_lines.txt'), 'r',
            encoding='latin-1') as f:
        for line_i in f:
            els = [el.strip() for el in line_i.split('+++$+++')]
            lines[els[0]] = {
                'line_id': els[0],
                'character_id': els[1],
                'movie_id': els[2],
                'character_name': els[3],
                'text': els[4]
            }
    return lines


def get_scripts(path='cornell movie-dialogs corpus'):
    """Summary

    Parameters
    ----------
    path : TYPE, optional
        Description

    Returns
    -------
    TYPE
        Description
    """
    lines = get_lines(path)
    conversations = get_conversations(path)
    script = []
    for conv_i in conversations:
        if len(conv_i['lines']) >= 2:
            for line_i in ast.literal_eval(conv_i['lines']):
                this_line = bs4.BeautifulSoup(lines[line_i]['text'],
                                              'lxml').text
                script.append(this_line)
    return script


def preprocess(text, min_count=10, max_length=40):
    """Summary

    Parameters
    ----------
    text : TYPE
        Description
    min_count : int, optional
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
    words = [[
        word.lower() for word in nltk.word_tokenize(s)
        if word not in string.punctuation
    ] for s in sentences]

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
    vocab = ['_PAD', '_GO', '_EOS', '_UNK'] + vocab
    # Then create the word to id mapping
    vocab = {k: v for v, k in enumerate(vocab)}

    with open('vocab.pkl', 'wb') as fp:
        pickle.dump(vocab, fp)

    unked = word2id(dataset, vocab)
    return unked, vocab


def word2id(words, vocab, UNK_ID=3):
    """Summary

    Parameters
    ----------
    words : TYPE
        Description
    vocab : TYPE
        Description
    UNK_ID : int, optional
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


def test_train():
    """Test training of cornell dataset with deprecated bucketed seq2seq model.
    """
    from cadl.deprecated import seq2seq_model as seq
    text = get_scripts()
    unked, vocab = preprocess(text)

    # Create bucketed pairs
    buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]
    pairs = {i: [] for i in range(len(buckets))}
    for pair_input, pair_output in zip(unked[:-1], unked[1:]):
        n_in, n_out = len(pair_input), len(pair_output)
        for bucket_i, bucket in enumerate(buckets):
            if n_in <= bucket[0] and n_out <= bucket[1]:
                pairs[bucket_i].append((pair_input[::-1], pair_output))
                break

    vocab_size = len(vocab)

    with tf.Session() as sess:
        net = seq.Seq2SeqModel(
            source_vocab_size=vocab_size,
            target_vocab_size=vocab_size,
            buckets=buckets,
            size=300,
            num_layers=3,
            max_gradient_norm=10.0,
            batch_size=64,
            learning_rate=0.0001,
            learning_rate_decay_factor=0.8,
            use_lstm=False)

        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        sess.run(init_op)

        n_iterations = 10000000
        lengths = [len(p) for p in pairs.values()]
        lengths = np.cumsum(lengths) / sum(lengths)
        for it_i in range(n_iterations):
            r = np.random.rand()
            bucket_id = 0
            while r > lengths[bucket_id]:
                bucket_id += 1
            encoder_inputs, decoder_inputs, target_weights = \
                net.get_batch(pairs, bucket_id)
            gradient_norm, perplexity, outputs = net.step(
                sess,
                encoder_inputs,
                decoder_inputs,
                target_weights,
                bucket_id,
                forward_only=False)
            print('{}: {}'.format(it_i, perplexity), end='\r')

            if it_i % 10000 == 0:
                net.saver.save(
                    sess, './seq2seq.ckpt', global_step=net.global_step)


def test_decode(sentence):
    """Test decoding of cornell dataset with deprecated seq2seq model.

    Parameters
    ----------
    sentence : TYPE
        Description
    """
    from cadl.deprecated import seq2seq_model as seq
    text = get_scripts()
    pairs, vocab, buckets = preprocess(text)
    vocab_size = len(vocab)

    with tf.Session() as sess:
        net = seq.Seq2SeqModel(
            source_vocab_size=vocab_size,
            target_vocab_size=vocab_size,
            buckets=buckets,
            size=300,
            num_layers=3,
            batch_size=1,
            max_gradient_norm=10.0,
            learning_rate=0.0001,
            learning_rate_decay_factor=0.8,
            forward_only=True,
            use_lstm=False)

        ckpt_path = tf.train.get_checkpoint_state('./').model_checkpoint_path
        net.saver.restore(sess, ckpt_path)

        def decode(sentence):
            """Summary

            Parameters
            ----------
            sentence : TYPE
                Description

            Returns
            -------
            TYPE
                Description
            """
            bucket_id = len(buckets) - 1
            preprocessed = [
                word
                for s in nltk.sent_tokenize(sentence.lower())
                for word in nltk.word_tokenize(s)
                if word not in string.punctuation
            ][::-1]
            if len(preprocessed) <= 0:
                return
            for b_i, b in enumerate(buckets):
                if b[0] >= len(preprocessed):
                    bucket_id = b_i
                    break
            tokens = word2id(preprocessed, vocab)
            encoder_inputs, decoder_inputs, target_weights = \
                net.get_batch({bucket_id: [(tokens[0], [])]}, bucket_id)
            _, _, output_logits = net.step(sess, encoder_inputs, decoder_inputs,
                                           target_weights, bucket_id, True)
            outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
            print(sentence, '\n', " ".join(id2word([outputs], vocab)[0]))


if __name__ == '__main__':
    test_train()
