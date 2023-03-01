from models import register_model
import torch.nn.functional as F
import torch.nn as nn
import torch
from models.base_model import BaseModel
from models.transformer_encoder_input import TransformerEncoderInput

@register_model("hidden_linear_wav_baseline")
class HiddenLinearWavBaseline(BaseModel):
    def __init__(self):
        super(HiddenLinearWavBaseline, self).__init__()

    def forward(self, inputs):
        hidden = F.relu(self.linear1(inputs))
        out = F.relu(self.linear_out(hidden))
        return out

    def build_model(self, cfg, input_dim):
        self.cfg = cfg
        self.linear1 = nn.Linear(in_features=input_dim, out_features=768)
        self.linear_out = nn.Linear(in_features=768, out_features=1) #TODO hardcode out_features
        #TODO hardcode in_features
