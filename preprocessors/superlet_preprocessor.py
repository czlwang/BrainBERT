import torch
import numpy as np
import torch.nn as nn
from scipy import signal, stats
from .superlet import superlet

class SuperletPreprocessor(nn.Module):
    def get_superlet(self, s_data, order_min=2, order_max=12, c_1=3, foi=None):
        #s_data is [.., n_samples]
        s_data = np.transpose(s_data)
        def scale_from_period(period):
            return period / (2 * np.pi)

        fs = 2048  # sampling frequency
        # frequencies of interest in Hz
        if foi is None:
            foi = np.linspace(5, 200, 50)
        scales = scale_from_period(1 / foi)

        spec = superlet(
            s_data,
            samplerate=fs,
            scales=scales,
            order_max=order_max,
            order_min=order_min,
            c_1=c_1,
            adaptive=True,
        )
        spec = np.abs(spec)
        decim = self.decim
        spec = spec[:,::decim]
        clip=5
        time = s_data.shape[0]/fs
        t = np.linspace(0,time,spec.shape[1])
        t = t[clip:-clip]
        spec = stats.zscore(spec[:,clip:-clip], axis=1)
        spec = spec.transpose(1,0)
        spec = np.nan_to_num(spec)
        spec = torch.FloatTensor(spec)
        return t, foi, spec

    def __init__(self, cfg):
        super(SuperletPreprocessor, self).__init__()
        self.cfg = cfg
        self.c1 = cfg.c1
        self.order_max = cfg.order_max
        self.order_min = cfg.order_min
        self.decim = cfg.decim

    def forward(self, wav):
        foi = np.linspace(self.cfg.min_f,self.cfg.max_f,self.cfg.n_f_steps)
        t, fs, spec = self.get_superlet(wav, order_min=self.order_min, order_max=self.order_max, c_1=self.c1, foi=foi)
        return spec
