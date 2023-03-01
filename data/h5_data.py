from dateutil import parser
import json
from datetime import datetime
import h5py
import os
from typing import Tuple, Dict, List
from types import SimpleNamespace

class H5Data():
    def __init__(self, subject: str, run_id: str, cfg) -> None:
        '''
        input:
            subject=subject id
        '''

        self.subject_id = subject
        self.samp_frequency = cfg.samp_frequency
        dataset_dir = cfg.raw_brain_data_dir
        trial = run_id

        # Path to neural data h5 file
        self.neural_data_file = os.path.join(dataset_dir,f'data-by-subject/{subject}/data/trials/{trial}/data-sharded.h5')

        # Path to neural data hash
        self.neural_data_hash = os.path.join(dataset_dir,f'data-by-subject/{subject}/data/trials/{trial}/data.h5.md5')

        # Path to experiment meta data directory
        self.headers_dir = os.path.join(dataset_dir, f'data-by-subject/{subject}/data/trials/{trial}/headers')

        # Path to file with timestamp
        self.timestamp_data = os.path.join(dataset_dir, f'data-by-subject/{subject}/data/trials/{trial}/time-stamp.json')

        # Path to brain regions csv file
        self.regions_file = os.path.join(dataset_dir, f'data-by-subject/{subject}/localization/depth-wm.csv')

        self.timestamp = self.get_timestamp()

    def get_timestamp(self):
        if not os.path.exists(self.timestamp_data):
            return None

        with open(self.timestamp_data, 'r') as f:
            d = json.load(f)
            timestamp = parser.parse(d["timestamp"])
            return timestamp

    def get_brain_region_localization(self) -> List[str]:
        '''
            returns list of electrodes in this subject and trial
            NOTE: the order of these labels is important. Their position corresponds with a row in data.h5
        '''
        def get_string_from_hdf5_reference(file, ref):
            return ''.join(chr(i) for i in file[ref[0]][:])

        header_file_name = os.listdir(self.headers_dir)[0]
        header_file = h5py.File(os.path.join(self.headers_dir, header_file_name), 'r')
        electrode_labels = [get_string_from_hdf5_reference(header_file, ref) for ref in header_file['channel_labels']]
        strip_name = lambda x: x.replace("*","").replace("#", "").replace("_", "")
        electrode_labels = [strip_name(e) for e in electrode_labels]
        return electrode_labels
