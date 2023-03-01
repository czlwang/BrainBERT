from .stft import STFTPreprocessor
from .morelet_preprocessor import MoreletPreprocessor
from .superlet_preprocessor import SuperletPreprocessor
from .wav_preprocessor import WavPreprocessor
from .superlet import superlet
from .spec_pretrained import SpecPretrained
from .spec_pooled import SpecPooled


__all__ = ["STFTPreprocessor",
           "MoreletPreprocessor",
           "SuperletPreprocessor",
           "superlet",
           "WavPreprocessor",
           "SpecPretrained",
           "SpecPooled"
          ]

def build_preprocessor(preprocessor_cfg):
    if preprocessor_cfg.name == "stft":
        extracter = STFTPreprocessor(preprocessor_cfg)
    elif preprocessor_cfg.name == "superlet":
        extracter = SuperletPreprocessor(preprocessor_cfg)
    elif preprocessor_cfg.name == "wav_preprocessor":
        extracter = WavPreprocessor(preprocessor_cfg)
    elif preprocessor_cfg.name == "spec_pretrained":
        extracter = SpecPretrained(preprocessor_cfg)
    elif preprocessor_cfg.name == "spec_pooled_preprocessor":
        extracter = SpecPooled(preprocessor_cfg)
    else:
        raise ValueError("Specify preprocessor")
    return extracter
