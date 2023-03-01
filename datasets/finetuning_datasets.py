import random
import os
import torch
from tqdm import tqdm as tqdm
import numpy as np
from omegaconf import DictConfig, OmegaConf
from torch.utils import data
from data.subject_data import SubjectData
from data.electrode_subject_data import ElectrodeSubjectData
from data.speech_nonspeech_subject_data import NonLinguisticSubjectData, SentenceOnsetSubjectData
from data.timestamped_subject_data import TimestampedSubjectData
from datasets import register_dataset
from preprocessors import build_preprocessor
from pathlib import Path
from .utils import save_cache
import logging

log = logging.getLogger(__name__)

class BaseFinetuning(data.Dataset):
    def __init__(self, cfg, preprocessor_cfg=None):
        super().__init__()

        self.cfg = cfg
        self.extracter = build_preprocessor(preprocessor_cfg)

        self.cache_input_features = None
        if "cache_input_features" in self.cfg:
            self.cache_input_features = self.cfg.cache_input_features

    def check_and_setup_cache(self):
        log.info("Checking for cache")
        reload_caches = self.cfg.get("reload_caches", False)
        if "cache_input_features" in self.cfg:
            if self.verify_cached_features(self.cfg.cache_input_features) and not reload_caches:
                log.info("Using cached input features")
            else:
                log.info("Saving new input features")
                self.save_cache(self.cfg.cache_input_features)
        else:
            log.info("No cached input features")

    def verify_cached_features(self, cfg):
        idxs = list(range(len(self)))
        random.shuffle(idxs)
        random_idxs = idxs[:3]
        log.info("Verifying cache")
        for idx in tqdm(random_idxs):
            cache_path = os.path.join(self.cache_input_features, f'{idx}.npy')
            try:
                cached = np.load(cache_path)
                live = self.get_source(idx, use_cache=False)[1].numpy()
                error = np.mean(np.abs(cached - live))
            except:
                return False
            if error > 1e-2:
                return False
        return True

    def get_source(self, idx, use_cache=True):
        wav = self.seeg_data[idx].astype('float32')
        if self.cache_input_features and use_cache:
            cache_path = os.path.join(self.cache_input_features, f'{idx}.npy')
            cached = np.load(cache_path)
            specs = torch.FloatTensor(cached)
        else:
            specs = self.extracter(wav)
        return len(specs), specs, wav

    def get_input_dim(self):
        item = self.__getitem__(0)
        return item["input"].shape[-1]

    def get_output_size(self):
        return 1 #single logit

    def __len__(self):
        return self.seeg_data.shape[0]

    def label2idx(self, label):
        return self.label2idx_dict[label]

    def save_cache(self, cache_path):
        Path(cache_path).mkdir(parents=True, exist_ok=True)
        save_cache(list(range(len(self))), cache_path, self.seeg_data, self.extracter)

        cfg_path = os.path.join("config.yaml")
        with open(cfg_path, 'w') as yamlfile:
            OmegaConf.save(config=self.cfg, f=yamlfile.name)
        
@register_dataset(name="speech_finetuning")
class SpeechFinetuning(BaseFinetuning):
    #NOTE: this is to be used in pre-training, while the other class in this file is to be used during fine-tuning
    def __init__(self, cfg, task_cfg=None, preprocessor_cfg=None) -> None:
        super().__init__(cfg, preprocessor_cfg=preprocessor_cfg)
        s = NonLinguisticSubjectData(cfg)

        self.word_df = s.labels
        self.seeg_data = s.neural_data
        assert len(self.cfg.electrodes) == 1
        assert self.seeg_data.shape[0] == 1
        self.seeg_data = self.seeg_data.squeeze(0)

        speech = set(self.word_df.linguistic_content)

        label2idx_dict = {}
        for idx, speech_id in enumerate(speech):
            label2idx_dict[speech_id] = idx
        self.label2idx_dict = label2idx_dict
        self.idx2label_dict = {k:v for v,k in label2idx_dict.items()}
        self.check_and_setup_cache()

    def __getitem__(self, idx: int):

        #NOTE: remember not to load to cuda here
        length, specs, wav = self.get_source(idx)
        return {
                "input" : specs,
                "length": length,
                "wav": wav,
                "label": self.label2idx(self.word_df.iloc[idx]['linguistic_content'])
               }

@register_dataset(name="movie_finetuning")
class MovieFineTuningDataset(BaseFinetuning):
    #NOTE: this is to be used in pre-training, while the other class in this file is to be used during fine-tuning
    def __init__(self, cfg, task_cfg=None) -> None:

        super().__init__(cfg)
        s = SubjectData(cfg)

        self.word_df = s.words
        self.seeg_data = s.neural_data
        assert len(self.cfg.electrodes) == 1
        assert self.seeg_data.shape[0] == 1
        self.seeg_data = self.seeg_data.squeeze(0)

        movie_ids = set(self.word_df.movie_id)

        label2idx_dict = {}
        for idx, movie in enumerate(movie_ids):
            label2idx_dict[movie] = idx
        self.label2idx_dict = label2idx_dict
        self.idx2label_dict = {k:v for v,k in label2idx_dict.items()}
        self.check_and_setup_cache() 

    def __getitem__(self, idx: int):

        #NOTE: remember not to load to cuda here
        length, specs, wav = self.get_source(idx)
        return {
                "input" : specs,
                "length": length,
                "wav": wav,
                "label": self.label2idx(self.word_df.iloc[idx]['movie_id'])
               }

@register_dataset(name="rms_finetuning")
class RMSFinetuning(BaseFinetuning):
    #NOTE: this is to be used in pre-training, while the other class in this file is to be used during fine-tuning
    def __init__(self, cfg, task_cfg=None, preprocessor_cfg=None) -> None:
        super().__init__(cfg, preprocessor_cfg=preprocessor_cfg)
        s = SubjectData(cfg)

        word_df = s.words
        seeg_data = s.neural_data
        assert len(cfg.electrodes) == 1
        assert seeg_data.shape[0] == 1
        seeg_data = seeg_data.squeeze(0)

        rms_mean = np.mean(word_df.rms)
        rms_std = np.std(word_df.rms)
        rms_idxs = (word_df.rms < (rms_mean - rms_std)) | (word_df.rms > (rms_mean + rms_std))
        rms_df = word_df[rms_idxs]

        rms_df["rms_bucket"] = rms_df.rms > rms_mean

        poss = set(rms_df.rms_bucket)
        self.word_df = rms_df
        self.seeg_data = seeg_data[rms_idxs]

        label2idx_dict = {}
        for idx, pos in enumerate(poss):
            label2idx_dict[pos] = idx
        self.label2idx_dict = label2idx_dict
        self.idx2label_dict = {k:v for v,k in label2idx_dict.items()}
        self.check_and_setup_cache()

    def __getitem__(self, idx: int):

        #NOTE: remember not to load to cuda here
        length, specs, wav = self.get_source(idx)
        return {
                "input" : specs,
                "length": length,
                "wav": wav,
                "label": self.label2idx(self.word_df.iloc[idx]['rms_bucket'])
               }

@register_dataset(name="brightness_finetuning")
class BrightnessFinetuning(BaseFinetuning):
    #NOTE: this is to be used in pre-training, while the other class in this file is to be used during fine-tuning
    def __init__(self, cfg, task_cfg=None, preprocessor_cfg=None) -> None:
        super().__init__(cfg, preprocessor_cfg=preprocessor_cfg)
        s = SubjectData(cfg)

        word_df = s.words
        seeg_data = s.neural_data
        assert len(cfg.electrodes) == 1
        assert seeg_data.shape[0] == 1
        seeg_data = seeg_data.squeeze(0)

        mean_pixel_brightness_mean = np.mean(word_df.max_mean_pixel_brightness)
        mean_pixel_brightness_std = np.std(word_df.max_mean_pixel_brightness)
        mean_pixel_brightness_idxs = (word_df.max_mean_pixel_brightness < (mean_pixel_brightness_mean - mean_pixel_brightness_std)) | (word_df.max_mean_pixel_brightness > (mean_pixel_brightness_mean + mean_pixel_brightness_std))
        mean_pixel_brightness_df = word_df[mean_pixel_brightness_idxs]

        mean_pixel_brightness_df["mean_pixel_brightness_bucket"] = mean_pixel_brightness_df.max_mean_pixel_brightness > mean_pixel_brightness_mean

        poss = set(mean_pixel_brightness_df.mean_pixel_brightness_bucket)
        self.word_df = mean_pixel_brightness_df
        self.seeg_data = seeg_data[mean_pixel_brightness_idxs]

        label2idx_dict = {}
        for idx, pos in enumerate(poss):
            label2idx_dict[pos] = idx
        self.label2idx_dict = label2idx_dict
        self.idx2label_dict = {k:v for v,k in label2idx_dict.items()}
        self.check_and_setup_cache()

    def __getitem__(self, idx: int):

        #NOTE: remember not to load to cuda here
        length, specs, wav = self.get_source(idx)
        return {
                "input" : specs,
                "length": length,
                "wav": wav,
                "label": self.label2idx(self.word_df.iloc[idx]['mean_pixel_brightness_bucket'])
               }

@register_dataset(name="pitch_finetuning")
class PitchFinetuning(BaseFinetuning):
    #NOTE: this is to be used in pre-training, while the other class in this file is to be used during fine-tuning
    def __init__(self, cfg, task_cfg=None, preprocessor_cfg=None) -> None:
        super().__init__(cfg, preprocessor_cfg=preprocessor_cfg)
        s = SubjectData(cfg)

        word_df = s.words
        seeg_data = s.neural_data
        assert len(cfg.electrodes) == 1
        assert seeg_data.shape[0] == 1
        seeg_data = seeg_data.squeeze(0)

        pitch_mean = np.mean(word_df.pitch)
        pitch_std = np.std(word_df.pitch)
        pitch_idxs = (word_df.pitch < (pitch_mean - pitch_std)) | (word_df.pitch > (pitch_mean + pitch_std))
        pitch_df = word_df[pitch_idxs]

        pitch_df["pitch_bucket"] = pitch_df.pitch > pitch_mean

        poss = set(pitch_df.pitch_bucket)
        self.word_df = pitch_df
        self.seeg_data = seeg_data[pitch_idxs]

        label2idx_dict = {}
        for idx, pos in enumerate(poss):
            label2idx_dict[pos] = idx
        self.label2idx_dict = label2idx_dict
        self.idx2label_dict = {k:v for v,k in label2idx_dict.items()}
        self.check_and_setup_cache()

    def __getitem__(self, idx: int):

        #NOTE: remember not to load to cuda here
        length, specs, wav = self.get_source(idx)
        return {
                "input" : specs,
                "length": length,
                "wav": wav,
                "label": self.label2idx(self.word_df.iloc[idx]['pitch_bucket'])
               }

@register_dataset(name="nounverb_finetuning")
class NounVerbFinetuning(BaseFinetuning):
    #NOTE: this is to be used in pre-training, while the other class in this file is to be used during fine-tuning
    def __init__(self, cfg, task_cfg=None, preprocessor_cfg=None) -> None:
        super().__init__(cfg, preprocessor_cfg=preprocessor_cfg)
        s = SubjectData(cfg)

        word_df = s.words
        seeg_data = s.neural_data
        assert len(cfg.electrodes) == 1
        assert seeg_data.shape[0] == 1
        seeg_data = seeg_data.squeeze(0)

        noun_verb_idxs = word_df.pos.isin(["NOUN","VERB"]).tolist()

        poss = set(word_df[noun_verb_idxs].pos)
        self.word_df = word_df[noun_verb_idxs]
        self.seeg_data = seeg_data[noun_verb_idxs]

        label2idx_dict = {}
        for idx, pos in enumerate(poss):
            label2idx_dict[pos] = idx
        self.label2idx_dict = label2idx_dict
        self.idx2label_dict = {k:v for v,k in label2idx_dict.items()}
        self.check_and_setup_cache()

    def __getitem__(self, idx: int):

        #NOTE: remember not to load to cuda here
        length, specs, wav = self.get_source(idx)
        return {
                "input" : specs,
                "length": length,
                "wav": wav,
                "label": self.label2idx(self.word_df.iloc[idx]['pos'])
               }

@register_dataset(name="finetuning_sentence_position")
class SentencePosition(BaseFinetuning):
    #NOTE: this is to be used in pre-training, while the other class in this file is to be used during fine-tuning
    def __init__(self, cfg, task_cfg=None, preprocessor_cfg=None) -> None:
        super().__init__(cfg, preprocessor_cfg=preprocessor_cfg)
        s = SubjectData(cfg)

        self.word_df = s.words
        self.seeg_data = s.neural_data
        assert len(self.cfg.electrodes) == 1
        assert self.seeg_data.shape[0] == 1
        self.seeg_data = self.seeg_data.squeeze(0)

        all_positions = set(s.words.idx_in_sentence)
        label2idx_dict = {}
        for idx, idx_in_sentence in enumerate(all_positions):
            label2idx_dict[idx_in_sentence] = idx
        self.label2idx_dict = label2idx_dict
        self.idx2label_dict = {k:v for v,k in label2idx_dict.items()}

        all_ordered_labels = s.trials[0].get_brain_region_localization()
        selected = [(i,e) for i,e in enumerate(all_ordered_labels) if e in cfg.electrodes]
        sel_idxs, sel_labels = zip(*selected)
        self.electrode_labels = list(sel_labels)
        assert len(self.electrode_labels) == len(self.cfg.electrodes)
 
        #self.check_and_setup_cache() 

    def __getitem__(self, idx: int):

        #NOTE: remember not to load to cuda here
        length, specs, wav = self.get_source(idx)
        return {
                "seeg_data" : specs,
                "length": length,
                "wav": wav,
                "label": self.label2idx(self.word_df.iloc[idx]['idx_in_sentence'])
               }

@register_dataset(name="onset_finetuning")
class OnsetFinetuning(BaseFinetuning):
    #NOTE: this is to be used in pre-training, while the other class in this file is to be used during fine-tuning
    def __init__(self, cfg, task_cfg=None, preprocessor_cfg=None) -> None:
        super().__init__(cfg, preprocessor_cfg=preprocessor_cfg)
        subj_data = SentenceOnsetSubjectData(cfg)

        self.word_df = subj_data.labels
        self.seeg_data = subj_data.neural_data
        assert len(self.cfg.electrodes) == 1
        assert self.seeg_data.shape[0] == 1
        self.seeg_data = self.seeg_data.squeeze(0)

        speech = set(self.word_df.linguistic_content)

        label2idx_dict = {}
        for idx, speech_id in enumerate(speech):
            label2idx_dict[speech_id] = idx
        self.label2idx_dict = label2idx_dict
        self.idx2label_dict = {k:v for v,k in label2idx_dict.items()}
        self.check_and_setup_cache() 

    def __getitem__(self, idx: int):

        #NOTE: remember not to load to cuda here
        length, specs, wav = self.get_source(idx)
        return {
                "input" : specs,
                "length": length,
                "wav": wav,
                "label": self.label2idx(self.word_df.iloc[idx]['linguistic_content'])
               }

@register_dataset(name="uniform_finetuning")
class UniformFinetuning(BaseFinetuning):
    #NOTE: this is to be used in pre-training, while the other class in this file is to be used during fine-tuning
    def __init__(self, cfg, task_cfg=None, preprocessor_cfg=None) -> None:

        super().__init__(cfg, preprocessor_cfg=preprocessor_cfg)
        subject_data = ElectrodeSubjectData(cfg.subject, cfg)
        self.subject_data = subject_data
        self.seeg_data = subject_data.neural_data
        #self.seeg_data = np.transpose(self.seeg_data, [1,0,2])
        assert len(cfg.electrodes) == 1
        assert self.seeg_data.shape[0] == 1
        self.seeg_data = self.seeg_data.squeeze(0)
        self.check_and_setup_cache()

    def __getitem__(self, idx: int):

        #NOTE: remember not to load to cuda here
        length, specs, wav = self.get_source(idx)
        return {
                "input" : specs,
                "length": length,
                "wav": wav,
                "label": 1, 
               }

@register_dataset(name="timestamp_finetuning")
class TimestampFinetuning(BaseFinetuning):
    #NOTE: this is to be used in pre-training, while the other class in this file is to be used during fine-tuning
    def __init__(self, cfg, task_cfg=None) -> None:

        super().__init__(cfg)
        subj_data = TimestampedSubjectData(cfg)

        timestamps = subj_data.labels
        seeg_data = subj_data.neural_data
        assert len(cfg.electrodes) == 1
        assert seeg_data.shape[0] == 1
        self.seeg_data = seeg_data.squeeze(0)
        self.timestamps = subj_data.labels

        hours = set([x.hour for x in self.timestamps])
        label2idx_dict = {}
        for idx, label in enumerate(hours):
            label2idx_dict[label] = idx
        self.label2idx_dict = label2idx_dict
        self.idx2label_dict = {k:v for v,k in label2idx_dict.items()}
        self.check_and_setup_cache()

    def __getitem__(self, idx: int):

        #NOTE: remember not to load to cuda here
        length, specs, wav = self.get_source(idx)
        return {
                "input" : specs,
                "length": length,
                "wav": wav,
                "label": self.label2idx(self.timestamps[idx].hour)
               }
