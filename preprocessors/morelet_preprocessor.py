import numpy as np
import torch
import torch.nn as nn
from scipy import signal, stats
import mne

class MoreletPreprocessor(nn.Module):
    def get_morelet(self, sig, fs, freqs, normalizing=None, **kwargs):

        sig = np.expand_dims(np.expand_dims(sig, 0), 0)
        arr = mne.time_frequency.tfr_array_morlet(sig, fs, freqs, decim=60)#NOTE decim hardcode
        arr = arr[0,0]
        morelet = stats.zscore(np.abs(arr)[:,10:-10], axis=1)
        morelet = morelet.transpose(1,0)
        return morelet

    def __init__(self):
        super(MoreletPreprocessor, self).__init__()

    def forward(self, wav):
        freqs = list(np.arange(10,200,4))
        morelet = self.get_morelet(wav, 2048, freqs) #TODO hardcode sampling rate
        return morelet
