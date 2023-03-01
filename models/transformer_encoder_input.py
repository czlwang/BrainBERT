import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        '''
        From https://discuss.pytorch.org/t/how-to-modify-the-positional-encoding-in-torch-nn-transformer/104308/2
        '''
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, seq):
        #seq is [batch, len, dim]
        assert len(seq.shape) == 3
        pos_enc = self.pe[:,:seq.size(1),:]
        out = seq + pos_enc
        test = torch.zeros_like(seq) + pos_enc
        return out, pos_enc

class TransformerEncoderInput(nn.Module):
    def __init__(self, cfg, dropout=0.1):
        super(TransformerEncoderInput, self).__init__()
        self.cfg = cfg
        self.in_proj = nn.Linear(in_features=cfg.input_dim, out_features=cfg.hidden_dim)
        self.positional_encoding = PositionalEncoding(self.cfg.hidden_dim)
        self.layer_norm = nn.LayerNorm(cfg.hidden_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, input_specs):
        input_specs = self.in_proj(input_specs)
        input_specs, pos_enc = self.positional_encoding(input_specs)
        input_specs = self.layer_norm(input_specs)
        input_specs = self.dropout(input_specs)
        return input_specs, pos_enc
