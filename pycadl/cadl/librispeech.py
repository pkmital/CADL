"""LibriSpeech dataset, batch processing, and preprocessing.
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
from scipy.io import wavfile
from cadl.utils import download_and_extract_tar
from glob import glob
import subprocess
import numpy as np


def get_dataset(saveto='librispeech', convert_to_wav=False, kind='dev'):
    """Download the LibriSpeech dataset and convert to wav files.

    More info: http://www.openslr.org/12/

    This interface downloads the LibriSpeech dataset and attempts to
    convert the flac to wave files using ffmpeg.  If you do not have ffmpeg
    installed, this function will not be able to convert the files to waves.

    Parameters
    ----------
    saveto : str
        Directory to save the resulting dataset ['librispeech']
    convert_to_wav : bool, optional
        Description
    kind : str, optional
        Description

    Returns
    -------
    TYPE
        Description
    """
    if not os.path.exists(saveto):
        if kind == 'dev':
            download_and_extract_tar(
                'http://www.openslr.org/resources/12/dev-clean.tar.gz', saveto)
        elif kind == 'train-100':
            download_and_extract_tar(
                'http://www.openslr.org/resources/12/train-clean-100.tar.gz',
                saveto)
        elif kind == 'train-360':
            download_and_extract_tar(
                'http://www.openslr.org/resources/12/train-clean-360.tar.gz',
                saveto)
        else:
            print('Not downloading.  Pass in either ["dev"],'
                  '"train-100", or "train-360", in order to '
                  'download the dataset.')

    wavs = glob('{}/**/*.wav'.format(saveto), recursive=True)
    if convert_to_wav:
        if len(wavs) == 0:
            flacs = glob('{}/**/*.flac'.format(saveto), recursive=True)
            for f in flacs:
                subprocess.check_call(
                    ['ffmpeg', '-i', f, '-f', 'wav', '-y', '%s.wav' % f])
            wavs = glob('{}/**/*.wav'.format(saveto), recursive=True)
        else:
            print('WARNING: Found existing wave files.  Not converting!')

    dataset = []
    for wav_i in wavs:
        id_i, chapter_i, utter_i = wav_i.split('/')[-3:]
        dataset.append({
            'name': wav_i,
            'id': id_i,
            'chapter': chapter_i,
            'utterance': utter_i.split('-')[-1].strip('.wav')
        })
    if len(wavs) == 0:
        print('LibriSpeech is a FLAC dataset.  Consider rerunning this '
              'command with convert_to_wav=True, to use ffmpeg to '
              'convert the flac files to wave files first. This requires '
              'the use of ffmpeg and so this should be installed first.')
    return dataset


def batch_generator(dataset,
                    batch_size=32,
                    max_sequence_length=6144,
                    maxval=32768.0,
                    threshold=0.2,
                    normalize=True):
    """Summary

    Parameters
    ----------
    dataset : TYPE
        Description
    batch_size : int, optional
        Description
    max_sequence_length : int, optional
        Description
    maxval : float, optional
        Description
    threshold : float, optional
        Description
    normalize : bool, optional
        Description

    Yields
    ------
    TYPE
        Description
    """
    n_batches = len(dataset) // batch_size
    for batch_i in range(n_batches):
        cropped_wavs, ids = [], []
        while len(cropped_wavs) < batch_size:
            idx_i = np.random.choice(np.arange(len(dataset)))
            fname_i = dataset[idx_i]['name']
            id_i = dataset[idx_i]['id']
            wav_i = wavfile.read(fname_i)[1]
            sample = np.random.choice(range(len(wav_i) - max_sequence_length))
            cropped_wav = wav_i[sample:sample + max_sequence_length]
            if np.max(np.abs(cropped_wav) / maxval) > threshold:
                if normalize:
                    cropped_wav = cropped_wav / maxval
                cropped_wavs.append(cropped_wav)
                ids.append(id_i)
        yield np.array(cropped_wavs, np.float32), np.array(ids, np.int32)
