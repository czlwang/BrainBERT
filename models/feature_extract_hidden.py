from models import register_model
import torch.nn.functional as F
import torch.nn as nn
import torch
from models.base_model import BaseModel
from models.transformer_encoder_input import TransformerEncoderInput

@register_model("feature_extract_hidden_model")
class FeatureExtractHiddenModel(BaseModel):
    def __init__(self):
        super(FeatureExtractHiddenModel, self).__init__()

    def forward(self, inputs):
        hidden = F.relu(self.linear_out1(inputs))
        out = self.linear_out(hidden)
        return out

    def build_model(self, cfg):
        self.cfg = cfg
        self.linear_out1 = nn.Linear(in_features=cfg.input_dim, out_features=50) 
        self.linear_out = nn.Linear(in_features=50, out_features=1) #TODO hardcode out_features

