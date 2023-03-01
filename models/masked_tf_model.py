from models import register_model
import torch.nn as nn
from models.base_model import BaseModel
from models.transformer_encoder_input import TransformerEncoderInput
from models.spec_prediction_head import SpecPredictionHead 

@register_model("masked_tf_model")
class MaskedTFModel(BaseModel):
    def __init__(self):
        super(MaskedTFModel, self).__init__()

    def forward(self, input_specs, src_key_mask, intermediate_rep=False, rep_from_layer=-1):
        input_specs, pos_enc = self.input_encoding(input_specs)
        input_specs = input_specs.transpose(0,1) #nn.Transformer wants [seq, batch, dim]
        if rep_from_layer==-1:
            output_specs = self.transformer(input_specs, src_key_padding_mask=src_key_mask)
        else:
            raise NotImplementedError
        output_specs = output_specs.transpose(0,1) #[batch, seq, dim]
        if intermediate_rep:
            return output_specs
        output_specs = self.spec_prediction_head(output_specs)
        return output_specs, pos_enc

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            if module.bias is not None:
                module.bias.data.zero_()
        if isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.bias.data.fill_(1.0)

    def build_model(self, cfg):
        self.cfg = cfg
        hidden_dim = self.cfg.hidden_dim
        self.input_encoding = TransformerEncoderInput(cfg)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=self.cfg.nhead, dim_feedforward=self.cfg.layer_dim_feedforward, activation=self.cfg.layer_activation)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.cfg.encoder_num_layers)
        self.spec_prediction_head = SpecPredictionHead(cfg)
        self.apply(self.init_weights)
