from torch import nn

class BaseCriterion(nn.Module):
    def __init__(self):
        super(BaseCriterion, self).__init__()
        pass

    def build_criterion(self, cfg):
        raise NotImplementedError

    def forward(self, model, batch, device):
        raise NotImplementedError
