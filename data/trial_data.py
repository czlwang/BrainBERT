import h5py
import os
import json
import pandas as pd
from typing import Tuple, Dict, List
from types import SimpleNamespace
import string
from .h5_data import H5Data

class TrialData(H5Data):
    def __init__(self, subject: str, trial, cfg) -> None:
        '''
        input:
            subject=subject id
            trial=trial id
            data_dir=path to ecog data
        '''
        super().__init__(subject, trial, cfg)
        self.trial_id = trial
        dataset_dir = cfg.raw_brain_data_dir

        # Path to trigger times csv file
        self.trigger_times_file = os.path.join(dataset_dir,f'data-by-subject/{subject}/data/trials/{trial}/trigger-times.csv')

        # Path to trial metadata json file
        self.metadata_file = os.path.join(dataset_dir,f'data-by-subject/{subject}/data/trials/{trial}/metadata.json')

        self.movie_id, _ = self.get_metadata()

        # Path to transcript csv file
        self.transcript_file = os.path.join(dataset_dir, f'transcripts/{self.movie_id}/manual/word-times-stanford.csv')

    def get_trigger_times(self) -> pd.DataFrame:
        '''
            returns the trigger times for this subject and trial
        '''
        trigs_df = pd.read_csv(self.trigger_times_file)
        return trigs_df

    def get_metadata(self) -> Tuple[str, Dict]:
        '''
            returns movie id and meta data dictionary
        '''
        with open(self.metadata_file, 'r') as f:
            meta_dict = json.load(f)
            movie_id = meta_dict['filename']
        return movie_id, meta_dict

    def get_movie_transcript(self) -> pd.DataFrame:
        '''
            returns dataframe of every word in the movie
            importantly, includes onset times for words
        '''
        words_df = pd.read_csv(self.transcript_file).set_index('Unnamed: 0')
        words_df = words_df.dropna().reset_index(drop=True)
        #words_df['text'] = list(map(str.lower, words_df['text']))
        #words_df['text'] = list(map(lambda s: s.translate(str.maketrans('', '', string.punctuation)), words_df['text']))
        return words_df
