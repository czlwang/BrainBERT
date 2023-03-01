from scipy import signal, stats#TODO remove import
import time
import os
import torch
import string
import numpy as np
import h5py
import logging
from pathlib import Path
# import numpy.typing as npt

from torch.utils import data
from .h5_data import H5Data
from .h5_data_reader import H5DataReader
from typing import Optional, List, Dict, Any, Tuple
import pandas as pd
from types import SimpleNamespace

log = logging.getLogger(__name__)

class ElectrodeSubjectData():
    def __init__(self, subject, cfg) -> None:
        self.selected_electrodes = cfg.electrodes
        self.cfg = cfg
        self.neural_data, self.trials = self.get_subj_data(subject)

    def save_cache(self, cached_data_path, seeg_data, subject_data_type):
        assert len(self.selected_electrodes)==1
        e = self.selected_electrodes[0]
        data_dir = self.make_cached_data_array_file_name(e, subject_data_type)
        e_path = os.path.join(cached_data_path, data_dir)
        Path(e_path).mkdir(exist_ok=True, parents=True)
        data_array_path = os.path.join(e_path, "array.npy")
        np.save(data_array_path, seeg_data)

    def make_cached_data_array_file_name(self, electrode, subject_data_type):
        cfg = self.cfg
        return f'cache_{cfg.duration}_d_{cfg.delta}_del_{cfg.rereference}_{cfg.subject}_{electrode}_{subject_data_type}_{cfg.high_gamma}_hg'

    def load_from_cache(self, cached_data_path, subject_data_type):
        assert len(self.cfg.electrodes)==1

        data_dir = self.make_cached_data_array_file_name(self.cfg.electrodes[0], subject_data_type)
        arr = None
        data_array_path = os.path.join(cached_data_path, data_dir, "array.npy")
        if os.path.exists(data_array_path):
            arr = np.load(data_array_path)

        return arr


    def get_subj_data(self, subject):
        '''
            returns:
                numpy array of words
                numpy array of shape [n_electrodes, n_words, n_samples] which holds the 
                    aligned data across all trials
        '''

        seeg_data, trials = [], []
        run_ids = self.cfg.brain_runs

        cached_data_path = self.cfg.get("cached_data_array", None)
        cache_name = "electrode_finetuning"
        reload_caches = self.cfg.get("reload_caches", False)
        use_cache = cached_data_path is not None and not reload_caches
        cache_exists = False
        if use_cache:
            data_dir = self.make_cached_data_array_file_name(self.cfg.electrodes[0], cache_name)
            data_array_path = os.path.join(cached_data_path, data_dir, "array.npy")
            cache_exists = os.path.exists(data_array_path)

        for run_id in run_ids:
            t = H5Data(subject, run_id, self.cfg)
            trials.append(t)
            reader = H5DataReader(t, self.cfg)

            log.info("Getting filtered data")
            if use_cache and cache_exists:
                continue
            else:
                seeg_trial_data = reader.get_filtered_data()
                seeg_data.append(seeg_trial_data)
        assert len(run_ids)==1
        if use_cache and cache_exists:
            seeg_data = self.load_from_cache(cached_data_path, cache_name)
        else:
            seeg_data = np.concatenate(seeg_data)
            cutoff_len = int(seeg_data.shape[-1] / (2048*self.cfg.duration))* 2048 * self.cfg.duration #how many 3 second samples should we take?
            cutoff_len = int(cutoff_len)
            seeg_data = seeg_data[:,:cutoff_len]
            seeg_data = seeg_data.reshape([seeg_data.shape[0],-1, int(2048*self.cfg.duration)]) #NOTE hardcode

        if cached_data_path is not None:
            self.save_cache(cached_data_path, seeg_data, cache_name)
        return seeg_data, trials
