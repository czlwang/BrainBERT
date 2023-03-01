import importlib
import os
from pathlib import Path

MODEL_REGISTRY = {}

__all__ = ["build_model"]

def build_model(cfg, *args, **kwargs):
    model_name = cfg.name
    assert model_name in MODEL_REGISTRY
    model = MODEL_REGISTRY[model_name]()
    model.build_model(cfg, *args, **kwargs)
    return model

def register_model(name):
    def register_model_cls(cls):
        if name in MODEL_REGISTRY:
            raise ValueError(f'{name} already in registry')
        else:
            MODEL_REGISTRY[name] = cls
        return cls
    return register_model_cls

def import_models():
    for file in os.listdir(os.path.dirname(__file__)):
        if file.endswith(".py") and not file.startswith("_"):
            module_name = str(Path(file).with_suffix(""))
            importlib.import_module('models.'+module_name)
import_models()
