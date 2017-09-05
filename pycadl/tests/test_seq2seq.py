from cadl import seq2seq, cornell
from numpy.testing import run_module_suite
import tensorflow as tf


def test_embedding_layer():
    with tf.Graph().as_default():
        vocab_size = 100
        embed_size = 20
        x = tf.placeholder(name='x', shape=[None, 1], dtype=tf.int32)
        embed_lookup, embed_matrix = seq2seq._create_embedding(x, vocab_size, embed_size)
        assert(embed_lookup.shape.as_list() == [None, 1, embed_size])
        assert(embed_matrix.shape.as_list() == [vocab_size, embed_size])

def test_rnn_cell():
    with tf.Graph().as_default():
        n_neurons = 5
        n_layers = 3
        keep_prob = 1.0
        cell = seq2seq._create_rnn_cell(n_neurons, n_layers, keep_prob)
        assert(cell.output_size == n_neurons)
        assert(len(cell.state_size) == n_layers)

def test_model():
    with tf.Graph().as_default():
        seq2seq.create_model()

def test_training():
    txt = cornell.get_scripts()
    with tf.Graph().as_default():
        seq2seq.train(txt[:20], batch_size=10, n_epochs=1)


if __name__ == "__main__":
    run_module_suite()
