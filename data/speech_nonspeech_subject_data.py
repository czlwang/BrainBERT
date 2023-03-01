from scipy import signal, stats#TODO remove import
import psutil
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

class NonLinguisticSubjectData():
    def __init__(self, cfg) -> None:
        self.selected_electrodes = cfg.electrodes
        self.cfg = cfg
        self.labels, self.neural_data, self.trials = self.get_subj_data(cfg.subject)

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

            duration = self.cfg.duration
            interval_duration = self.cfg.interval_duration
            seeg_trial_no_word_data, labels = reader.get_aligned_non_words_matrix(duration=duration, interval_duration=interval_duration)
            labels['movie_id'] = t.movie_id
            trials.append(t)
            words.append(labels)
            seeg_data.append(seeg_trial_no_word_data)

        neural_data = np.concatenate(seeg_data, axis=1)
        labels_df = pd.concat(words) #NOTE the index will not be unique, but the location will
        #TODO: pretty sure we are missing the get_subj_data method here
        return labels_df, neural_data, trials

class SentenceOnsetSubjectData():
    def __init__(self, cfg) -> None:
        self.selected_electrodes = cfg.electrodes
        self.cfg = cfg
        self.labels, self.neural_data, self.trials = self.get_subj_data(cfg.subject)

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

            duration = self.cfg.duration
            delta = self.cfg.delta
            interval_duration = self.cfg.interval_duration
            seeg_trial_no_word_data, labels = reader.get_aligned_speech_onset_matrix(duration=duration, interval_duration=interval_duration)
            labels['movie_id'] = t.movie_id
            trials.append(t)
            words.append(labels)
            seeg_data.append(seeg_trial_no_word_data)

        neural_data = np.concatenate(seeg_data, axis=1)
        labels_df = pd.concat(words) #NOTE the index will not be unique, but the location will
        #TODO: pretty sure we are missing the get_subj_data method here
        return labels_df, neural_data, trials
