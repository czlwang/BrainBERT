from models import register_model
import torch.nn as nn
import torch
from models.base_model import BaseModel
from models.transformer_encoder_input import TransformerEncoderInput

@register_model("linear_wav_baseline")
class LinearWavModel(BaseModel):
    def __init__(self):
        super(LinearWavModel, self).__init__()

    def forward(self, inputs):
        out = self.linear_out(inputs)
        return out

    def build_model(self, cfg, input_dim):
        self.cfg = cfg
        self.linear_out = nn.Linear(in_features=input_dim, out_features=1) #TODO hardcode out_features
        #TODO hardcode in_features
