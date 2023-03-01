import torch.nn as nn

class SpecPredictionHead(nn.Module): 
    def __init__(self, cfg):
        super(SpecPredictionHead, self).__init__()
        self.hidden_layer = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        self.act_fn = None
        if cfg.layer_activation=="gelu":
            self.act_fn = nn.GELU()
        self.layer_norm = nn.LayerNorm(cfg.hidden_dim)
        self.output = nn.Linear(cfg.hidden_dim, cfg.input_dim)

    def forward(self, hidden):
        h = self.hidden_layer(hidden)
        h = self.act_fn(h)
        h = self.layer_norm(h)
        h = self.output(h)
        return h
