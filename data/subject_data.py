from scipy import signal, stats#TODO remove import
import time
import os
import torch
import string
import numpy as np
import h5py
# import numpy.typing as npt

from torch.utils import data
from .trial_data import TrialData
from .trial_data_reader import TrialDataReader
from typing import Optional, List, Dict, Any, Tuple
import pandas as pd
from types import SimpleNamespace

class SubjectData():
    def __init__(self, cfg) -> None:
        self.selected_electrodes = cfg.electrodes
        self.selected_words = cfg.words
        self.cfg = cfg
        self.words, self.neural_data, self.trials = self.get_subj_data(cfg.subject)

    def get_subj_data(self, subject):
        words, seeg_data, trials = [], [], []
        cached_transcript_aligns = self.cfg.cached_transcript_aligns
        for trial in self.cfg.brain_runs:
            if cached_transcript_aligns: #TODO: I want to make this automatic
                cached_transcript_aligns = os.path.join(cached_transcript_aligns, subject, trial)
                os.makedirs(cached_transcript_aligns, exist_ok=True)
                self.cfg.cached_transcript_aligns = cached_transcript_aligns
            t = TrialData(subject, trial, self.cfg)
            reader = TrialDataReader(t, self.cfg)

            trial_words, seeg_trial_data = reader.get_aligned_predictor_matrix(duration=self.cfg.duration, delta=self.cfg.delta)
            assert (range(seeg_trial_data.shape[1]) == trial_words.index).all()
            trial_words['movie_id'] = t.movie_id
            trials.append(t)
            words.append(trial_words)
            seeg_data.append(seeg_trial_data)

        neural_data = np.concatenate(seeg_data, axis=1)
        #neural_data is [n_electrodes, n_words, n_samples]
        words_df = pd.concat(words) #NOTE the index will not be unique, but the location will
        return words_df, neural_data, trials
