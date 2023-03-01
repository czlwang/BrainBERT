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

class SpecPretrained(nn.Module):
    def __init__(self, cfg):
        super(SpecPretrained, self).__init__()
        self.spec_preprocessor = build_preprocessor(cfg.spec_name, cfg)

        self.cfg = cfg
        ckpt_path = cfg.upstream_ckpt
        init_state = torch.load(ckpt_path)
        upstream_cfg = init_state["model_cfg"]
        if upstream_cfg.name=='debug_model':
            upstream_cfg.name='masked_tf_model'
        self.upstream = models.build_model(upstream_cfg)
        #model.module.load_weights(states)
        states = init_state["model"]
        self.upstream.load_weights(states)

    def forward(self, wav, spec_preprocessed=None):
        if spec_preprocessed is None:
            spec = self.spec_preprocessor(wav)
        else:
            spec = torch.FloatTensor(spec_preprocessed)
        inputs = spec.unsqueeze(0) #[batch, time, num_freq_channels]
        pad_mask = torch.zeros(1, spec.shape[0], dtype=bool)
        self.upstream.eval()
        middle = int(inputs.shape[1]/2)
        with torch.no_grad():
            #clip=50
            #outputs = self.upstream(inputs[:,middle-clip:middle+clip], pad_mask[:, middle-clip:middle+clip], intermediate_rep=True)
            rep_from_layer = -1
            if "rep_from_layer" in self.cfg:
                rep_from_layer = self.cfg.rep_from_layer 
            outputs = self.upstream(inputs, pad_mask, intermediate_rep=True, rep_from_layer=rep_from_layer)
        middle = int(outputs.shape[1]/2)
        out = outputs[:,middle-5:middle+5].mean(axis=1)
        #out = outputs.mean(axis=1)
        out = out.squeeze(0)
        return out
