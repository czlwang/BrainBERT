from .base_criterion import BaseCriterion
import torch
from torch import nn
from criterions import register_criterion

@register_criterion("seeg_wav2vec_criterion")
class SeegWav2VecCriterion(BaseCriterion):
    def __init__(self):
        super(SeegWav2VecCriterion, self).__init__()
        pass

    def build_criterion(self, cfg):
        self.cfg = cfg

    def forward(self, model, batch, device):
        #x = batch[:,:10] #potentially don't move to device if dataparallel
        print(batch)
        import pdb; pdb.set_trace()
        return loss, logging_output
