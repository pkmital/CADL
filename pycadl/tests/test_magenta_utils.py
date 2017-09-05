import os
from cadl import magenta_utils
from cadl.utils import exists, download
from numpy.testing import run_module_suite
from magenta.music import midi_io


def test_harry_potter_exists():
    assert(exists('https://s3.amazonaws.com/cadl/share/27334_Harry-Potter-Theme-Monophonic.mid'))
    assert(exists('https://s3.amazonaws.com/cadl/share/21150_Harry-Potter-and-the-Philosophers-Stone.mid'))


def test_basic_rnn_mag_exists():
    assert(exists('https://s3.amazonaws.com/cadl/models/basic_rnn.mag'))


def test_harry_potter():
    primer_midi = download('https://s3.amazonaws.com/cadl/share/21150_Harry-Potter-and-the-Philosophers-Stone.mid')
    primer_sequence = midi_io.midi_file_to_sequence_proto(primer_midi)
    assert(len(primer_sequence.notes) == 282)
    assert(len(primer_sequence.time_signatures) == 1)


def test_convert_to_monophonic():
    primer_midi = download('https://s3.amazonaws.com/cadl/share/21150_Harry-Potter-and-the-Philosophers-Stone.mid')
    primer_sequence = midi_io.midi_file_to_sequence_proto(primer_midi)
    magenta_utils.convert_to_monophonic(primer_sequence)
    assert(len(primer_sequence.notes) == 254)


def test_melody_rnn_generation():
    primer_midi = download('https://s3.amazonaws.com/cadl/share/21150_Harry-Potter-and-the-Philosophers-Stone.mid')
    download('https://s3.amazonaws.com/cadl/models/basic_rnn.mag')

    magenta_utils.synthesize(primer_midi)
    fname = 'primer.mid'
    assert(os.path.exists('primer.mid'))
    generated_primer = midi_io.midi_file_to_sequence_proto(fname)
    assert(len(generated_primer.notes) == 14)

    fname = 'synthesis.mid'
    assert(os.path.exists('synthesis.mid'))
    generated_synthesis = midi_io.midi_file_to_sequence_proto(fname)
    assert(len(generated_synthesis.notes) == 243)

if __name__ == "__main__":
    run_module_suite()
