"""Various utilities for working with Google's Magenta Library.
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
import os
import tempfile
from collections import defaultdict
import tensorflow as tf
import pretty_midi
import magenta
from magenta.music import midi_io, sequences_lib
from magenta.music.melodies_lib import Melody
from magenta.models.melody_rnn import melody_rnn_sequence_generator, \
    melody_rnn_model
from magenta.protobuf import generator_pb2


def convert_to_monophonic(seq):
    """Summary

    Parameters
    ----------
    seq : TYPE
        Description
    """
    note_i = 1
    while note_i < len(seq.notes):
        n_prev = seq.notes[note_i - 1]
        n_curr = seq.notes[note_i]
        if n_curr.start_time >= n_prev.start_time and \
                n_curr.start_time < n_prev.end_time:
            seq.notes.remove(n_prev)
        else:
            note_i += 1


def parse_midi_file(midi_file,
                    max_notes=float('Inf'),
                    max_time_signatures=1,
                    max_tempos=1,
                    ignore_polyphonic_notes=True,
                    convert_to_drums=False,
                    steps_per_quarter=16):
    """Summary

    Parameters
    ----------
    midi_file : TYPE
        Description
    max_notes : TYPE, optional
        Description
    max_time_signatures : int, optional
        Description
    max_tempos : int, optional
        Description
    ignore_polyphonic_notes : bool, optional
        Description
    convert_to_drums : bool, optional
        Description
    steps_per_quarter : int, optional
        Description

    Returns
    -------
    TYPE
        Description
    """
    seq = midi_io.midi_file_to_sequence_proto(midi_file)

    while len(seq.notes) > max_notes:
        seq.notes.pop()

    while len(seq.time_signatures) > max_time_signatures:
        seq.time_signatures.pop()

    while len(seq.tempos) > max_tempos:
        seq.tempos.pop()

    if convert_to_drums:
        for note_i in range(len(seq.notes)):
            seq.notes[note_i].program = 10

    if ignore_polyphonic_notes:
        convert_to_monophonic(seq)

    seq = sequences_lib.quantize_note_sequence(
        seq, steps_per_quarter=steps_per_quarter)

    if seq.tempos:
        qpm = seq.tempos[0].qpm
    else:
        qpm = 120

    melody = Melody()
    melody.from_quantized_sequence(
        seq, ignore_polyphonic_notes=ignore_polyphonic_notes)
    seq = melody.to_sequence(qpm=qpm)

    return seq, qpm


def parse_midi_dataset(directory, saveto='parsed', **kwargs):
    """Summary

    Parameters
    ----------
    directory : TYPE
        Description
    saveto : str, optional
        Description
    **kwargs
        Description

    Returns
    -------
    TYPE
        Description
    """
    if not os.path.exists(saveto):
        os.mkdir(saveto)
    failed = []
    for root, dirs, files in os.walk(directory):
        for name in files:
            if name.endswith('mid'):
                fname = os.path.join(root, name)
                seq, qpm = parse_midi_file(fname, **kwargs)
                out_fname = os.path.join(saveto, name)
                magenta.music.sequence_proto_to_midi_file(seq, out_fname)
    return failed


def synthesize(midi_file, model='basic', num_steps=2000,
               max_primer_notes=32, temperature=1.0,
               beam_size=1, branch_factor=1,
               steps_per_quarter=16, **kwargs):
    """Summary

    Parameters
    ----------
    midi_file : TYPE
        Description
    model : str, optional
        Description
    num_steps : int, optional
        Description
    max_primer_notes : int, optional
        Description
    temperature : float, optional
        Description
    beam_size : int, optional
        Description
    branch_factor : int, optional
        Description
    steps_per_quarter : int, optional
        Description
    **kwargs
        Description
    """
    config = melody_rnn_model.default_configs['{}_rnn'.format(model)]
    bundle_file = '{}_rnn.mag'.format(model)
    generator = melody_rnn_sequence_generator.MelodyRnnSequenceGenerator(
        model=melody_rnn_model.MelodyRnnModel(config),
        details=config.details,
        steps_per_quarter=steps_per_quarter,
        bundle=magenta.music.read_bundle_file(bundle_file)
    )

    # Get a protobuf of the MIDI file
    seq, qpm = parse_midi_file(midi_file, **kwargs)

    opt = generator_pb2.GeneratorOptions()
    seconds_per_step = 60.0 / qpm / steps_per_quarter
    total_seconds = num_steps * seconds_per_step
    last_end_time = max(n.end_time for n in seq.notes)
    opt.generate_sections.add(
        start_time=last_end_time + seconds_per_step,
        end_time=total_seconds)
    opt.args['temperature'].float_value = temperature
    opt.args['beam_size'].int_value = beam_size
    opt.args['branch_factor'].int_value = branch_factor
    opt.args['steps_per_iteration'].int_value = 1

    print(opt)
    generated = generator.generate(seq, opt)
    fname = 'primer.mid'
    magenta.music.sequence_proto_to_midi_file(seq, fname)
    fname = 'synthesis.mid'
    magenta.music.sequence_proto_to_midi_file(generated, fname)


def sequence_proto_to_pretty_midi(
        sequence, is_drum=False,
        drop_events_n_seconds_after_last_note=None):
    """Convert tensorflow.magenta.NoteSequence proto to a PrettyMIDI.

    Time is stored in the NoteSequence in absolute values (seconds) as opposed to
    relative values (MIDI ticks). When the NoteSequence is translated back to
    PrettyMIDI the absolute time is retained. The tempo map is also recreated.

    Parameters
    ----------
    sequence
        A tensorfow.magenta.NoteSequence proto.
    is_drum : bool, optional
        Description
    drop_events_n_seconds_after_last_note
        Events (e.g., time signature changes)
        that occur this many seconds after the last note will be dropped. If
        None, then no events will be dropped.

    Returns
    -------
    A pretty_midi.PrettyMIDI object or None if sequence could not be decoded.
    """
    from magenta.music import constants
    _PRETTY_MIDI_MAJOR_TO_MINOR_OFFSET = 12

    ticks_per_quarter = (sequence.ticks_per_quarter
                         if sequence.ticks_per_quarter else
                         constants.STANDARD_PPQ)

    max_event_time = None
    if drop_events_n_seconds_after_last_note is not None:
        max_event_time = (max([n.end_time for n in sequence.notes]) +
                          drop_events_n_seconds_after_last_note)

    # Try to find a tempo at time zero. The list is not guaranteed to be in
    # order.
    initial_seq_tempo = None
    for seq_tempo in sequence.tempos:
        if seq_tempo.time == 0:
            initial_seq_tempo = seq_tempo
            break

    kwargs = {}
    kwargs['initial_tempo'] = (initial_seq_tempo.qpm if initial_seq_tempo
                               else constants.DEFAULT_QUARTERS_PER_MINUTE)
    pm = pretty_midi.PrettyMIDI(resolution=ticks_per_quarter, **kwargs)

    # Create an empty instrument to contain time and key signatures.
    instrument = pretty_midi.Instrument(0, is_drum=is_drum)
    pm.instruments.append(instrument)

    # Populate time signatures.
    for seq_ts in sequence.time_signatures:
        if max_event_time and seq_ts.time > max_event_time:
            continue
        time_signature = pretty_midi.containers.TimeSignature(
            seq_ts.numerator, seq_ts.denominator, seq_ts.time)
        pm.time_signature_changes.append(time_signature)

    # Populate key signatures.
    for seq_key in sequence.key_signatures:
        if max_event_time and seq_key.time > max_event_time:
            continue
        key_number = seq_key.key
        if seq_key.mode == seq_key.MINOR:
            key_number += _PRETTY_MIDI_MAJOR_TO_MINOR_OFFSET
        key_signature = pretty_midi.containers.KeySignature(
            key_number, seq_key.time)
        pm.key_signature_changes.append(key_signature)

    # Populate tempos.
    # TODO(douglaseck): Update this code if pretty_midi adds the ability to
    # write tempo.
    for seq_tempo in sequence.tempos:
        # Skip if this tempo was added in the PrettyMIDI constructor.
        if seq_tempo == initial_seq_tempo:
            continue
        if max_event_time and seq_tempo.time > max_event_time:
            continue
        tick_scale = 60.0 / (pm.resolution * seq_tempo.qpm)
        tick = pm.time_to_tick(seq_tempo.time)
        # pylint: disable=protected-access
        pm._tick_scales.append((tick, tick_scale))
        pm._update_tick_to_time(0)
        # pylint: enable=protected-access

    # Populate instrument events by first gathering notes and other event types
    # in lists then write them sorted to the PrettyMidi object.
    instrument_events = defaultdict(lambda: defaultdict(list))
    for seq_note in sequence.notes:
        instrument_events[(seq_note.instrument, seq_note.program,
                           seq_note.is_drum)]['notes'].append(
            pretty_midi.Note(
                seq_note.velocity, seq_note.pitch,
                seq_note.start_time, seq_note.end_time))
    for seq_bend in sequence.pitch_bends:
        if max_event_time and seq_bend.time > max_event_time:
            continue
        instrument_events[(seq_bend.instrument, seq_bend.program,
                           seq_bend.is_drum)]['bends'].append(
            pretty_midi.PitchBend(seq_bend.bend, seq_bend.time))
    for seq_cc in sequence.control_changes:
        if max_event_time and seq_cc.time > max_event_time:
            continue
        instrument_events[(seq_cc.instrument, seq_cc.program,
                           seq_cc.is_drum)]['controls'].append(
            pretty_midi.ControlChange(
                seq_cc.control_number,
                seq_cc.control_value, seq_cc.time))

    for (instr_id, prog_id, is_drum) in sorted(instrument_events.keys()):
        # For instr_id 0 append to the instrument created above.
        if instr_id > 0:
            instrument = pretty_midi.Instrument(prog_id, is_drum)
            pm.instruments.append(instrument)
        instrument.program = prog_id
        instrument.notes = instrument_events[
            (instr_id, prog_id, is_drum)]['notes']
        instrument.pitch_bends = instrument_events[
            (instr_id, prog_id, is_drum)]['bends']
        instrument.control_changes = instrument_events[
            (instr_id, prog_id, is_drum)]['controls']

    return pm


def sequence_proto_to_midi_file(sequence, output_file,
                                is_drum=False,
                                drop_events_n_seconds_after_last_note=None):
    """Convert tensorflow.magenta.NoteSequence proto to a MIDI file on disk.

    Time is stored in the NoteSequence in absolute values (seconds) as opposed to
    relative values (MIDI ticks). When the NoteSequence is translated back to
    MIDI the absolute time is retained. The tempo map is also recreated.

    Parameters
    ----------
    sequence
        A tensorfow.magenta.NoteSequence proto.
    output_file
        String path to MIDI file that will be written.
    is_drum : bool, optional
        Description
    drop_events_n_seconds_after_last_note
        Events (e.g., time signature changes)
        that occur this many seconds after the last note will be dropped. If
        None, then no events will be dropped.
    """
    pretty_midi_object = sequence_proto_to_pretty_midi(
        sequence, is_drum, drop_events_n_seconds_after_last_note)
    with tempfile.NamedTemporaryFile() as temp_file:
        pretty_midi_object.write(temp_file.name)
        tf.gfile.Copy(temp_file.name, output_file, overwrite=True)
