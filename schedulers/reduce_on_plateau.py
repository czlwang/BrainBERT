import torch
from .base_scheduler import BaseScheduler

class ReduceOnPlateau(BaseScheduler):
    def __init__(self, cfg, optim):
        super(ReduceOnPlateau, self).__init__()

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, 'min', patience=300)
        self.scheduler.step(100) #TODO hack

    def step(self, loss):
        self.scheduler.step(loss)
