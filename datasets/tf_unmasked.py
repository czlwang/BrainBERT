import torch
import random
from torch.utils import data 
import os
import numpy as np
from datasets import register_dataset
from preprocessors import STFTPreprocessor, MoreletPreprocessor, SuperletPreprocessor

@register_dataset(name="tf_unmasked")
#This is supposed to look like the datasets in finetuning.py
class TFUnmasked(data.Dataset):
    def __init__(self, cfg, task_cfg=None, preprocessor_cfg=None):
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

        if preprocessor_cfg.name == "stft":
            extracter = STFTPreprocessor()
        elif preprocessor_cfg.name == 'morelet':
            extracter = MoreletPreprocessor()
        elif preprocessor_cfg.name == 'superlet':
            extracter = SuperletPreprocessor(preprocessor_cfg)
        else:
            raise RuntimeError("Specify a preprocessor")
        self.extracter = extracter

    def get_input_dim(self):
        item = self.__getitem__(0)
        return item["input"].shape[-1]

    def __len__(self):
        return len(self.lengths)

    def __getitem__(self, idx):
        file_name = self.files[idx]
        file_name = os.path.join(self.root_dir, file_name)
        data = np.load(file_name)

        data = data.astype('float32')
        #rand_len = random.randrange(1000, len(data), 1)
        rand_len = -1
        wav = data[:rand_len]
        data = self.extracter(wav)

        return {"input": data,
                "label": 1}

