"""TEDLium Dataset.
"""
"""
SPH format info
---------------
Channels            : 1
Sample Rate     : 16000
Precision           : 16-bit
Bit Rate            : 256k
Sample Encoding : 16-bit Signed Integer PCM

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


def get_dataset():
    """Summary

    Returns
    -------
    TYPE
        Description
    """
    stms = []
    for dirpath, dirnames, filenames in os.walk('TEDLIUM_release2'):
        for f in filenames:
            if f.endswith('stm'):
                stms.append(os.path.join(dirpath, f))

    data = []
    for stm_i in stms:
        with open(stm_i, 'r') as fp:
            lines = fp.readlines()
        for line_i in lines:
            sp = line_i.split()
            data.append({
                'id': sp[0],
                'num': sp[1],
                'id2': sp[2],
                'start_time': sp[3],
                'end_time': sp[4],
                'ch': 'wideband' if 'f0' in sp[5] else 'telephone',
                'sex': 'male' if 'male' in sp[5] else 'female',
                'text': " ".join(
                    sp[6:]) if sp[6] != 'ignore_time_segment_in_scoring' else ''})

    for max_duration in range(30):
        durations = []
        for stm_i in stms:
            with open(stm_i, 'r') as fp:
                lines = fp.readlines()
            for line_i in lines:
                sp = line_i.split()
                dur = float(sp[4]) - float(sp[3])
                if dur < max_duration:
                    durations.append(dur)

    return data, durations
