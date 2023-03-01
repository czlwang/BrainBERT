import importlib
import os
from pathlib import Path

TASK_REGISTRY = {}

__all__ = ["setup_task"]

def setup_task(cfg):
    task_name = cfg.name
    assert task_name in TASK_REGISTRY
    task = TASK_REGISTRY[task_name]
    return task.setup_task(cfg)

def register_task(name):
    def register_task_cls(cls):
        if name in TASK_REGISTRY:
            raise ValueError(f'{name} already in registry')
        else:
            TASK_REGISTRY[name] = cls
        return cls
    return register_task_cls

def import_tasks():
    for file in os.listdir(os.path.dirname(__file__)):
        if file.endswith(".py") and not file.startswith("_"):
            module_name = str(Path(file).with_suffix(""))
            importlib.import_module('tasks.'+module_name)

import_tasks()
