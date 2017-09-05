"""VCTK Dataset download and preprocessing.
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


def get_dataset(saveto='vctk', convert_to_16khz=False):
    """Download the VCTK dataset and convert to wav files.

    More info:
        http://homepages.inf.ed.ac.uk/jyamagis/
        page3/page58/page58.html

    This interface downloads the VCTK dataset and attempts to
    convert the flac to wave files using ffmpeg.  If you do not have ffmpeg
    installed, this function will not be able to convert the files to waves.

    Parameters
    ----------
    saveto : str
        Directory to save the resulting dataset ['vctk']
    convert_to_16khz : bool, optional
        Description

    Returns
    -------
    TYPE
        Description
    """
    if not os.path.exists(saveto):
        download_and_extract_tar(
            'http://homepages.inf.ed.ac.uk/jyamagis/' +
            'release/VCTK-Corpus.tar.gz',
            saveto)

    wavs = glob('{}/**/*.16khz.wav'.format(saveto), recursive=True)
    if convert_to_16khz and len(wavs) == 0:
        wavs = glob('{}/**/*.wav'.format(saveto), recursive=True)
        for f in wavs:
            subprocess.check_call(
                ['ffmpeg', '-i', f, '-f', 'wav', '-ar', '16000', '-y', '%s.16khz.wav' % f])

    wavs = glob('{}/**/*.16khz.wav'.format(saveto), recursive=True)

    dataset = []
    for wav_i in wavs:
        chapter_i, utter_i = wav_i.split('/')[-2:]
        dataset.append({
            'name': wav_i,
            'chapter': chapter_i,
            'utterance': utter_i.split('-')[-1].strip('.wav')})
    return dataset


def batch_generator(dataset, batch_size=32, max_sequence_length=6144,
                    maxval=32768.0, threshold=0.2, normalize=True):
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
        cropped_wavs = []
        while len(cropped_wavs) < batch_size:
            idx_i = np.random.choice(np.arange(len(dataset)))
            fname_i = dataset[idx_i]['name']
            wav_i = wavfile.read(fname_i)[1]
            if len(wav_i) > max_sequence_length:
                sample = np.random.choice(range(len(wav_i) - max_sequence_length))
                cropped_wav = wav_i[sample:sample + max_sequence_length]
                if np.max(np.abs(cropped_wav) / maxval) > threshold:
                    if normalize:
                        cropped_wav = cropped_wav / maxval
                    cropped_wavs.append(cropped_wav)
        yield np.array(cropped_wavs, np.float32)
