from .stft import STFTPreprocessor
from .morelet_preprocessor import MoreletPreprocessor
from .superlet_preprocessor import SuperletPreprocessor
import torch
import torch.nn as nn
import models
import os

#This preprocssor combines a spectrogram preprocessor with a feature extracter (transformer)

def build_preprocessor(spec_name, preprocessor_cfg):
    if spec_name == "stft":
        extracter = STFTPreprocessor(preprocessor_cfg)
    elif spec_name == "superlet":
        extracter = SuperletPreprocessor(preprocessor_cfg)
    return extracter

class SpecPooled(nn.Module):
    def __init__(self, cfg):
        super(SpecPooled, self).__init__()
        self.spec_preprocessor = build_preprocessor(cfg.spec_name, cfg)

    def forward(self, wav, spec_preprocessed=None):
        if spec_preprocessed is None:
            spec = self.spec_preprocessor(wav)
        else:
            spec = torch.FloatTensor(spec_preprocessed)
        inputs = spec.unsqueeze(0) #[batch, time, num_freq_channels]
        outputs = inputs
        middle = int(outputs.shape[1]/2)
        out = outputs[:,middle-5:middle+5].mean(axis=1)
        #out = outputs.mean(axis=1)
        out = out.squeeze(0)
        return out
