from datetime import timedelta
from scipy import signal, stats#TODO remove import
import psutil
import time
import pytz
import os
import torch
import string
import numpy as np
import h5py
# import numpy.typing as npt

from torch.utils import data
from .h5_data import H5Data
from .h5_data_reader import H5DataReader
from typing import Optional, List, Dict, Any, Tuple
import pandas as pd
from types import SimpleNamespace

class TimestampedSubjectData():
    def __init__(self, cfg) -> None:
        self.selected_electrodes = cfg.electrodes
        self.selected_words = cfg.words
        self.cfg = cfg
        self.neural_data, self.trials, self.labels = self.get_subj_data(cfg.subject)

        #convert everything to US east coast time
        est = pytz.timezone('US/Eastern')
        self.labels = [t.astimezone(est) for t in self.labels]

        #Only get the night hours for a single day
        days = np.array([x.day for x in self.labels])
        hours = np.array([x.hour for x in self.labels])
        single_day_idxs = days==(days[0]+1) #hand selected day
        night_idxs = (hours >= 1) & (hours <= 5)
        single_night_idxs = (night_idxs) & (single_day_idxs)
        night_samples = np.array(self.labels)[single_night_idxs]
        self.labels = night_samples
        assert self.neural_data.shape[0]==1
        self.neural_data = self.neural_data[:,single_night_idxs]

    def get_subj_data(self, subject):
        seeg_data, trials, timestamps = [], [], []
        for trial in self.cfg.brain_runs:
            t = H5Data(subject, trial, self.cfg)
            reader = H5DataReader(t, self.cfg)

            timestamp = t.get_timestamp()
            seeg_trial_data = reader.get_filtered_data()
            trials.append(t)
            duration = self.cfg.duration

            cutoff_len = int(seeg_trial_data.shape[-1] / (2048*duration))* 2048 * duration #how many samples should we take?
            cutoff_len = int(cutoff_len)
            seeg_trial_data = seeg_trial_data[:,:cutoff_len]
            seeg_trial_data = seeg_trial_data.reshape([seeg_trial_data.shape[0],-1, int(2048*duration)]) #NOTE hardcode
            trial_timestamps = [timestamp + timedelta(seconds=int(duration*i)) for i in range(seeg_trial_data.shape[1])]

            timestamps += trial_timestamps
            seeg_data.append(seeg_trial_data)

        assert len(self.cfg.brain_runs)==1
        seeg_data = np.concatenate(seeg_data)
        return seeg_data, trials, timestamps
