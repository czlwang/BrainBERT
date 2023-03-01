from .reduce_on_plateau import ReduceOnPlateau
from .ramp_up import RampUp

__all__ = ["build_scheduler"]

def build_scheduler(cfg, optim):
    name = cfg.name
    if name=="reduce_on_plateau":
        return ReduceOnPlateau(cfg, optim)
    if name=="ramp_up":
        return RampUp(cfg, optim)
    else:
        raise ValueError("Scheduler name not found")
