import torch
import random
from torch.utils import data 
import os
import numpy as np
from scipy.io import wavfile
from datasets import register_dataset
from preprocessors import STFTPreprocessor

class BaseTFDataset(data.Dataset):
    #Parent time-frequency dataset
    def __init__(self, cfg, task_cfg=None):
        extracter = DebugPreprocessor()
        self.cfg = cfg
        self.task_cfg = task_cfg
        manifest_path = cfg.data
        manifest_path = os.path.join(manifest_path, "train.tsv")
        with open(manifest_path, "r") as f:
            lines = f.readlines() 
        self.root_dir = lines[0].strip()
        files, lengths = [], []
        for x in lines[1:]:
            row = x.strip().split('\t')
            files.append(row[0])
            lengths.append(row[1])
        self.files, self.lengths = files, lengths
        self.extracter = extracter

    def get_input_dim(self):
        item = self.__getitem__(0)
        return item["masked_input"].shape[-1]

    def mask_time(self, data):
        mask_label = torch.zeros_like(data)

        consecutive_min = self.task_cfg.time_mask_consecutive_min
        consecutive_max = self.task_cfg.time_mask_consecutive_max
        assert consecutive_min <= consecutive_max
        assert consecutive_max < data.shape[0]
        valid_starts = range(len(data)-consecutive_max)
        masked_steps = [i for i in valid_starts if random.random() < self.task_cfg.time_mask_p]
        masked_steps = [(i, i+random.randint(consecutive_min, consecutive_max)) for i in masked_steps]

        for (start,end) in masked_steps:
            mask_label[start:end,:] = 1

        masked_data = torch.clone(data)
        for (start,end) in masked_steps:
            if random.random() < 0.85: #NOTE hardcode dice
                masked_data[start:end,:] = 0

        return masked_data, mask_label

    def __len__(self):
        return len(self.lengths)

    def __getitem__(self, idx):
        file_name = self.files[idx]
        file_name = os.path.join(self.root_dir, file_name)
        #raw_wave = np.load(file_name)
        samplerate, data = wavfile.read(file_name)

        data = data.astype('float32')
        #rand_len = random.randrange(1000, len(data), 1)
        rand_len = -1
        data = data[:rand_len]
        data = self.extracter(data)

        masked_data, mask_label = self.mask_time(data) 

        return {"masked_input": masked_data,
                "length": data.shape[0],
                "mask_label": mask_label,
                "target": data}

