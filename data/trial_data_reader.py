import random
from scipy.signal import hilbert, chirp
from tqdm import tqdm
import os
import h5py
import numpy as np
import pandas as pd
import scipy.stats
import logging
# import numpy.typing as npt

from typing import Optional, List, Tuple
from .trial_data import TrialData
from scipy import signal, stats
from .utils import compute_m5_hash
from .h5_data_reader import H5DataReader
from pathlib import Path

log = logging.getLogger(__name__)

class TrialDataReader(H5DataReader):
    def __init__(self, trial_data, cfg) -> None:
        '''
            Input: trial_data=ecog and word data to perform processing on
        '''
        super().__init__(trial_data, cfg)

        self.start_col = 'start'
        self.end_col = 'end'
        self.trig_time_col = 'movie_time'
        self.trig_idx_col = 'index'
        self.est_idx_col = 'est_idx'
        self.est_end_idx_col = 'est_end_idx'
        self.word_time_col = 'word_time'
        self.word_text_col = 'text'
        self.is_onset_col = 'is_onset'
        self.is_offset_col = 'is_offset'

        cached_transcript_aligns = cfg.cached_transcript_aligns
        self.aligned_script_df = self.get_aligned_movie_transcript(cached_transcript_aligns)

        self.selected_words = [] #TODO hardcode

    def estimate_sample_index(self, t, near_t, near_trig):
        '''
            input: movie time t and the closest trigger time
            returns: linear interpolation to the nearest sample index to t
        '''
        samp_frequency = self.trial_data.samp_frequency
        trig_diff = (t-near_t)*samp_frequency
        return round(near_trig+trig_diff)

    def add_estimated_sample_index(self, w_df: pd.DataFrame) -> pd.DataFrame:
        '''
            input: a dataframe of word features
            returns: the input word dataframe, but augmented with onset and offset times
        '''
        tmp_w_df = w_df.copy(deep=True)
        trigs_df = self.trial_data.get_trigger_times()
        last_t = trigs_df.loc[len(trigs_df) - 1, self.trig_time_col]

        for i, t, endt in tqdm(zip(w_df.index, w_df[self.start_col], w_df[self.end_col])):
            if t > last_t:
                break
            idx = (abs(trigs_df[self.trig_time_col] - t)).idxmin()
            tmp_w_df.loc[i, :] = w_df.loc[i, :]
            trigger_t = trigs_df.loc[idx, self.trig_time_col]
            trigger_idx = trigs_df.loc[idx, self.trig_idx_col] 
            tmp_w_df.loc[i, self.est_idx_col] = self.estimate_sample_index(t, trigger_t, trigger_idx)

            end_idx = (abs(trigs_df[self.trig_time_col] - endt)).idxmin()
            end_trigger_t = trigs_df.loc[end_idx, self.trig_time_col]
            end_trigger_idx = trigs_df.loc[end_idx, self.trig_idx_col] 
            tmp_w_df.loc[i, self.est_end_idx_col] = self.estimate_sample_index(endt, end_trigger_t, end_trigger_idx)
        return tmp_w_df

    def get_aligned_movie_transcript(self, cached_transcript_aligns: str) -> pd.DataFrame:
        '''
            returns the dataframe of word data for the trial, but augmented with onset and offset times
        '''

        save_path = None
        if cached_transcript_aligns:
            save_path = os.path.join(cached_transcript_aligns, "aligned_script.h5") 
            computed_hash = compute_m5_hash(self.trial_data.transcript_file)

        if save_path and os.path.exists(save_path): 
            cached_df = pd.read_hdf(save_path, key='transcript_data')
            if cached_df['orig_transcript_hash'][0] == computed_hash:
                return cached_df

        words_df = self.trial_data.get_movie_transcript()
        words_df = self.add_estimated_sample_index(words_df)

        if save_path:
            words_df['orig_transcript_hash'] = computed_hash #TODO eventually look into storing metadata
            words_df.to_hdf(save_path, key='transcript_data')

        return words_df

    def select_words(self, words_df: pd.DataFrame) -> pd.DataFrame:
        '''
        Input:
            word_window_arr = array of shape [n_electrodes, n_words, n_samples]
            words_df = pandas dataframe of word features 
        Output:
            pandas dataframe of the word data where only
                rows with the selected words are present
            
        '''
        filtered_df = words_df[words_df['text'] != '']

        if self.selected_words==[]:
            return filtered_df

        filtered_df = filtered_df[filtered_df['text'].isin(self.selected_words)]
        return filtered_df

    def make_cached_data_array_file_name(self, electrode, subject_data_type):
        cfg = self.cfg
        return f'cache_{cfg.duration}_d_{cfg.delta}_del_{cfg.rereference}_{cfg.subject}_{electrode}_{subject_data_type}_{cfg.high_gamma}_hg'
  
    def save_cache(self, cached_data_path, results, labels_df, subject_data_type):
        electrodes = self.get_ordered_electrodes(self.cfg.electrodes)
        for i,e in enumerate(electrodes):
            arr = np.expand_dims(results[i], axis=0)
            data_dir = self.make_cached_data_array_file_name(e, subject_data_type)
            e_path = os.path.join(cached_data_path, data_dir)
            Path(e_path).mkdir(exist_ok=True, parents=True)
            data_array_path = os.path.join(e_path, "array.npy")
            np.save(data_array_path, arr)
            data_label_path = os.path.join(e_path, "labels.csv")
            labels_df.to_csv(data_label_path)

    def load_from_cache(self, cached_data_path, subject_data_type):
        if len(self.cfg.electrodes) > 1:
            return None, None

        data_dir = self.make_cached_data_array_file_name(self.cfg.electrodes[0], subject_data_type)
        labels = None
        data_label_path = os.path.join(cached_data_path, data_dir, "labels.csv")
        if os.path.exists(data_label_path):
            labels = pd.read_csv(data_label_path)
        arr = None
        data_array_path = os.path.join(cached_data_path, data_dir, "array.npy")
        if os.path.exists(data_array_path):
            arr = np.load(data_array_path)

        return arr, labels

    def get_aligned_linguistic_control_matrix(self, duration: int=3, interval_duration=None, onsets_only=True):
        '''
            input:
                duration=context to return as input in seconds. For example, if duration=5s, then 
                    5s worth of data will be returned
                interval_duration=intervals to divide the movie into. For example, if
                    interval_duration=1s, the movie will be divided into 1s chunks, and each chunk
                    will be returned embedded in 5s worth of context.
            output:
                an array of shape [n_electrodes, n_words, n_samples]
                for each electrode, each row is the ecog data from self.trial_data for a given word
        '''
        if interval_duration is None:
            interval_duration = duration
        log.info("Getting onset and non-speech intervals")
        cached_data_path = self.cfg.get("cached_data_array", None)
        cache_name = "onset_finetuning" if onsets_only else "speech_finetuning"
        reload_caches = self.cfg.get("reload_caches", False)
        if cached_data_path is not None and not reload_caches:
            arr, labels = self.load_from_cache(cached_data_path, cache_name)
            if labels is not None and arr is not None:
                return arr, labels

        filtered_data = self.get_filtered_data()

        w_df = self.aligned_script_df
        w_df = self.select_words(w_df)

        samp_frequency = self.trial_data.samp_frequency
        input_window_duration = int(duration*samp_frequency)
        input_left, input_right = int(input_window_duration/2), input_window_duration-int(input_window_duration/2) 

        interval_window_duration = int(interval_duration*samp_frequency)

        w_df = w_df.iloc[w_df[self.est_idx_col].dropna().index] #drop all rows that don't have a start time

        start_idxs = w_df[self.est_idx_col].astype(int).tolist()
        end_idxs = w_df[self.est_end_idx_col].astype(int).tolist()
        word_intervals = list(zip(start_idxs, end_idxs))
        total_length = filtered_data.shape[-1]
        total_n_intervals = int(total_length/interval_window_duration)
        all_intervals = [(i*interval_window_duration, (i+1)*interval_window_duration) for i in range(total_n_intervals)]
        intersect = lambda a, b: a[0] < b[1] and a[1] > b[0]
        intersect_with_word = lambda a: np.any([intersect(a,b) for b in word_intervals])
        non_word_intervals = list(filter(lambda x: not intersect_with_word(x), all_intervals))

        if onsets_only:
            w_df = w_df[w_df.is_onset.astype(bool)]

        start_idxs = w_df[self.est_idx_col].astype(int)
        start_idxs = (start_idxs - input_left).astype(int)
        end_idxs = (start_idxs + input_window_duration).astype(int)
        word_intervals = list(zip(start_idxs.tolist(), end_idxs.tolist()))

        valid_non_word_intervals = []
        for (start, end) in non_word_intervals:
            center = int((start+end)/2)
            if center-input_window_duration>0 and center+input_window_duration<filtered_data.shape[1]:
                valid_non_word_intervals.append((center-input_left, center+input_right))

        all_non_word_samples = np.stack([filtered_data[:, start:end] for (start,end) in valid_non_word_intervals])
        all_word_samples = np.stack([filtered_data[:, start:end] for (start,end) in word_intervals])

        balanced_len = min(len(all_word_samples), len(all_non_word_samples))
        random.seed(42)
        non_word_idxs = list(range(len(all_non_word_samples)))
        non_word_idxs = random.sample(non_word_idxs, balanced_len)
        all_non_word_samples = all_non_word_samples[non_word_idxs]
        word_idxs = list(range(len(all_word_samples)))
        word_idxs = random.sample(word_idxs, balanced_len)
        all_word_samples = all_word_samples[word_idxs]

        result = np.concatenate([all_word_samples, all_non_word_samples], axis=0)
        labels = np.repeat([True, False], [all_word_samples.shape[0], all_non_word_samples.shape[0]])
        result = np.transpose(result, [1,0,2]) #[n_electrodes, n_intervals, n_samples]
        labels_df = pd.DataFrame({"linguistic_content": labels})

        if cached_data_path is not None:
            self.save_cache(cached_data_path, result, labels_df, cache_name)
        return result, labels_df

    def get_aligned_non_words_matrix(self, duration: int=3, interval_duration=None):
        return self.get_aligned_linguistic_control_matrix(duration, interval_duration=interval_duration, onsets_only=False)

    def get_aligned_speech_onset_matrix(self, duration: int=3, interval_duration=None):
        return self.get_aligned_linguistic_control_matrix(duration, interval_duration=interval_duration, onsets_only=True)

    def get_aligned_predictor_matrix(self, duration: int=3, delta: int=-1, save_path: Optional[str]=None) -> Tuple[pd.DataFrame, np.ndarray]:
        '''
            input:
                delta=context to take before onset
                duration=context to take after start. Note that the start = onset + delta.
                save_path=where to save/load the aligned matrix
            output:
                an array of shape [n_electrodes, n_words, n_samples]
                for each electrode, each row is the ecog data from self.trial_data for a given word
        '''
        cached_data_path = self.cfg.get("cached_data_array", None)
        reload_caches = self.cfg.get("reload_caches", False)
        if cached_data_path is not None and not reload_caches:
            print("cached_data_path", cached_data_path)
            try:
                arr, labels = self.load_from_cache(cached_data_path, "subject-data")
            except:
                import pdb; pdb.set_trace()
            if labels is not None and arr is not None:
                return labels, arr

        filtered_data = self.get_filtered_data()

        w_df = self.aligned_script_df
        w_df = self.select_words(w_df)

        samp_frequency = self.trial_data.samp_frequency
        window_duration = int(duration*samp_frequency)
        window_onset = int(delta*samp_frequency)

        w_df = w_df.iloc[w_df[self.est_idx_col].dropna().index] #drop all rows that don't have a start time
        word_window_arr = np.empty((filtered_data.shape[0], len(w_df.index), window_duration))

        log.info('Generating aligned predictor matrix of size {}'.format((filtered_data.shape[0], len(w_df.index), window_duration)))
        start_idxs = w_df[self.est_idx_col].astype(int) + window_onset
        end_idxs = start_idxs + window_duration
        for i,word_idx in tqdm(enumerate(w_df.index)):
            row = w_df.loc[word_idx]
            try:
                start = start_idxs[word_idx]
                end = end_idxs[word_idx]
                word_window_arr[:, i, :] = filtered_data[:, start:end]
            except ValueError as err:
                print('Neural recording stopped before movie ended')
                break
        w_df['ecog_idx'] = range(word_window_arr.shape[1])
        if cached_data_path is not None:
            self.save_cache(cached_data_path, word_window_arr, w_df, "subject-data")
        return w_df, word_window_arr #TODO check that this w_df is the one being used at train time  
