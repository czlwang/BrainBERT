from glob import glob as glob
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
        self.neural_data_file = os.path.join(dataset_dir, f'all_subject_data/{subject}_{trial}.h5')

        # Path to brain regions csv file
        self.regions_file = os.path.join(dataset_dir, f'localization/{subject}/depth-wm.csv')

        electrode_labels_file = glob(os.path.join(dataset_dir, "electrode_labels", subject, "electrode_labels.json"))
        assert len(electrode_labels_file)==1
        electrode_labels_file = electrode_labels_file[0]
        self.electrode_labels_file = electrode_labels_file
        self.timestamp_data = os.path.join(dataset_dir, "timestamps", f"{subject}_{trial}_timestamp.json")

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
        with open(self.electrode_labels_file, "r") as f:
            electrode_labels = json.load(f)
        strip_string = lambda x: x.replace("*","").replace("#","").replace("_","")
        electrode_labels = [strip_string(e) for e in electrode_labels]
        return electrode_labels
