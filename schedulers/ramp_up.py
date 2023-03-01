import torch
from .base_scheduler import BaseScheduler
from torch.optim.lr_scheduler import StepLR, ExponentialLR
from torch.optim.sgd import SGD
from warmup_scheduler import GradualWarmupScheduler

class RampUp(BaseScheduler):
    def __init__(self, cfg, optim):
        '''
        https://github.com/ildoonet/pytorch-gradual-warmup-lr
        '''
        super(RampUp, self).__init__()
        self.cfg = cfg
        warmup = int(self.cfg.warmup*self.cfg.total_steps)
        step_size = (self.cfg.total_steps - warmup)/100
        scheduler_steplr = StepLR(optim, step_size=step_size, gamma=0.99)
        scheduler_warmup = GradualWarmupScheduler(optim, multiplier=1, total_epoch=warmup, after_scheduler=scheduler_steplr)

        # this zero gradient update is needed to avoid a warning message, issue #8.
        optim.zero_grad()
        optim.step()
        self.scheduler = scheduler_warmup

    def step(self, loss):
        self.scheduler.step()
