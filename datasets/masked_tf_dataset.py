from omegaconf import OmegaConf
import torch
import random
from torch.utils import data 
import os
import numpy as np
from scipy.io import wavfile
from datasets import register_dataset
from preprocessors import STFTPreprocessor
from util.mask_utils import mask_inputs

@register_dataset(name="masked_tf_dataset")
class MaskedTFDataset(data.Dataset):
    def __init__(self, cfg, task_cfg=None, preprocessor_cfg=None):
        #THE PLAN
        #also make masked_tf_datased_from_cached
        self.cfg = cfg
        self.task_cfg = task_cfg
        manifest_path = cfg.data
        manifest_path = os.path.join(manifest_path, "manifest.tsv")
        with open(manifest_path, "r") as f:
            lines = f.readlines() 
        self.root_dir = lines[0].strip()
        files, lengths = [], []
        for x in lines[1:]:
            row = x.strip().split('\t')
            files.append(row[0])
            lengths.append(row[1])
        self.files, self.lengths = files, lengths

        self.cached_features = None

        if 'cached_features' in cfg:
            self.cached_features = cfg.cached_features
            self.initialize_cached_features(cfg.cached_features)
        elif preprocessor_cfg.name=="stft":
            extracter = STFTPreprocessor(preprocessor_cfg)
            self.extracter = extracter
        else:
            raise RuntimeError("Specify preprocessor")

    def initialize_cached_features(self, cache_root):
        cfg_path = os.path.join(cache_root, "config.yaml")
        loaded = OmegaConf.load(cfg_path)
        assert self.cfg.preprocessor == loaded.data.preprocessor

        manifest_path = os.path.join(cache_root, "manifest.tsv")
        with open(manifest_path, "r") as f:
            lines = f.readlines() 
        self.cache_root_dir = lines[0].strip()
        orig2cached = {} #Map original file to cached feature file
        for x in lines[2:]:
            row = x.strip().split('\t')
            orig2cached[row[0]] = row[1]
        self.orig2cached = orig2cached

    def get_input_dim(self):
        item = self.__getitem__(0)
        return item["masked_input"].shape[-1]


    def __len__(self):
        return len(self.lengths)

    def get_cached_features(self, file_name):
        file_name = self.orig2cached[file_name] 
        file_name = os.path.join(self.cache_root_dir, file_name)
        data = np.load(file_name)
        data = np.nan_to_num(data) #For superlet caches
        data = torch.FloatTensor(data)
        return data
        
    def __getitem__(self, idx):
        file_name = self.files[idx]
        file_path = os.path.join(self.root_dir, file_name)
        data = np.load(file_path)

        data = data.astype('float32')
        #rand_len = random.randrange(1000, len(data), 1)
        rand_len = -1
        wav = data[:rand_len]

        if self.cached_features:
            data = self.get_cached_features(file_name)
        else:
            data = self.extracter(wav)

        masked_data, mask_label = mask_inputs(data, self.task_cfg) 
        return {"masked_input": masked_data,
                "length": data.shape[0],
                "mask_label": mask_label,
                "wav": wav,
                "target": data}
