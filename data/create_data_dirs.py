#example:
#python3 -m data.create_data_dirs +data=pretraining +hydra.job.chdir=False
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_original_cwd, to_absolute_path
import logging
from datasets import build_dataset
from tqdm import tqdm as tqdm
import os
from scipy.io.wavfile import write
from pathlib import Path
import numpy as np

log = logging.getLogger(__name__)

def print_time(start):
    return ((time.time() - start)/60)

def write_dataset_to_dir(dataset, args, root_out):
    subject_name = args.subject
    run_name = args.brain_runs[0]
    subject_out = os.path.join(root_out, subject_name, run_name) 
    Path(subject_out).mkdir(parents=True, exist_ok=True)

    trainable = 0
    samplerate = args.samp_frequency
    for i in tqdm(range(len(dataset))):
        file_name = os.path.join(subject_out, f'{i}.wav')
        data = dataset[i]
        data = data["input"]
        write(file_name, samplerate, data.astype(np.float64))

@hydra.main(config_path="../conf", version_base=None)
def main(cfg: DictConfig) -> None:
    log.info("create data dir")
    log.info(OmegaConf.to_yaml(cfg, resolve=True))
    print(HydraConfig.get().job.name)
    print(HydraConfig.get().run.dir)
    out_dir = HydraConfig.get().run.dir
    out_dir = to_absolute_path(out_dir)
    log.info(f'output directory {out_dir}')

    dataset = build_dataset(cfg.data)
    write_dataset_to_dir(dataset, cfg.data, out_dir)

if __name__ == "__main__":
    main()
