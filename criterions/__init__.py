import importlib
import os
from pathlib import Path

CRITERION_REGISTRY = {}

__all__ = ["build_criterion"]

def build_criterion(cfg):
    criterion_name = cfg.name
    assert criterion_name in CRITERION_REGISTRY
    criterion = CRITERION_REGISTRY[criterion_name]()
    criterion.build_criterion(cfg)
    return criterion

def register_criterion(name):
    def register_criterion_cls(cls):
        if name in CRITERION_REGISTRY:
            raise ValueError(f'{name} already in registry')
        else:
            CRITERION_REGISTRY[name] = cls
        return cls
    return register_criterion_cls

def import_criterions():
    for file in os.listdir(os.path.dirname(__file__)):
        if file.endswith(".py") and not file.startswith("_"):
            module_name = str(Path(file).with_suffix(""))
            importlib.import_module('criterions.'+module_name)

import_criterions()
