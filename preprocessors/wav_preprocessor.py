import numpy as np
import torch
import torch.nn as nn
from scipy import signal, stats

class WavPreprocessor(nn.Module):
    def __init__(self, cfg):
        super(WavPreprocessor, self).__init__()
        self.cfg = cfg

    def forward(self, wav):
        middle = int(len(wav)/2)
        sr = self.cfg.sample_rate
        if "clip_seconds" in self.cfg:
            clip_window = int(sr*self.cfg.clip_seconds)
            assert clip_window*2 < len(wav)
            return wav[middle-clip_window:middle+clip_window]
        return wav
