from torch.utils import data 
import numpy as np
from datasets import register_dataset
import os

@register_dataset(name="raw_wav_file_dataset")
class RawWavFileDataset(data.Dataset):
    def __init__(self, cfg, task_cfg=None, preprocessor_cfg=None):
        self.cfg = cfg
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

    def get_input_dim(self):
        item = self.__getitem__(0)
        return item["input"].shape[-1]

    def __len__(self):
        return len(self.lengths)

    def __getitem__(self, idx):
        file_name = self.files[idx]
        file_name = os.path.join(self.root_dir, file_name)
        data = np.load(file_name)

        wav = data.astype('float32')

        return {"input": wav}
