from data.subject_data import SubjectData
from datasets import register_dataset
from .finetuning_datasets import BaseFinetuning
import pandas as pd
import numpy as np

@register_dataset(name="single_subject_all_electrode")
class SingleSubjectAllElectrode(BaseFinetuning):
    def __init__(self, cfg, task_cfg=None, preprocessor_cfg=None) -> None:

        super().__init__(cfg, preprocessor_cfg=preprocessor_cfg)
        s = SubjectData(cfg)
        self.regions_file = s.trials[0].regions_file
        self.regions = pd.read_csv(self.regions_file)

        self.word_df = s.words
        self.seeg_data = s.neural_data
        assert len(self.cfg.electrodes) == self.seeg_data.shape[0]

        if cfg.onsets_only:
            onset_idxs = self.word_df.loc[self.word_df.is_onset.astype(bool)].index.tolist()
            self.word_df = self.word_df.loc[onset_idxs]
            self.seeg_data = self.seeg_data[:,onset_idxs]
            
        all_ordered_labels = s.trials[0].get_brain_region_localization()
        selected = [(i,e) for i,e in enumerate(all_ordered_labels) if e in cfg.electrodes]
        sel_idxs, sel_labels = zip(*selected)
        self.electrode_labels = list(sel_labels)
        assert len(self.electrode_labels) == len(self.cfg.electrodes)

    def get_source(self, idx, use_cache=True):
        wavs = self.seeg_data[:,idx].astype('float32') # a matrix of size [n_electrodes, n_samples]
        specs = []
        for j in range(wavs.shape[0]):
            wav = wavs[j]
            specs.append(self.extracter(wav))
        specs = np.stack(specs)
        length = specs.shape[1]
        return length, specs, wavs

    def __getitem__(self, idx):
        sentence_activities = self.seeg_data[:,idx,:]
        length, specs, wav = self.get_source(idx)

        all_sentence_data = {
            "labels": self.electrode_labels,
            "seeg_data": specs
        }
        return all_sentence_data

    def __len__(self):
        return self.seeg_data.shape[1]
