#to write features to disk
# python3 -m data.modify_manifest +data=pretrain_wavs_from_disk ++data.data=pretrain_data/manifests +data_prep=remove_subjects ++data_prep.out_dir=/storage/czw/self_supervised_seeg/pretrain_data/manifests_no_subject3/ +preprocessor=wav_preprocessor
from multiprocessing import Process, Queue
from omegaconf import DictConfig, OmegaConf
import hydra
import models
import tasks
from datasets import build_dataset
import logging
import os
from pathlib import Path
import numpy as np
from tqdm import tqdm as tqdm
import csv
import yaml
import json

#take dataset with wavs and write it to cache
log = logging.getLogger(__name__)

def write_manifest(root_dir, out_dir, cfg):
    #map the original file to the cached feature
    manifest_path = os.path.join(root_dir, "manifests")
    Path(manifest_path).mkdir(parents=True, exist_ok=True)
    old_manifest_path = os.path.join(manifest_path, "manifest.tsv")
    old_rows = []
    with open(old_manifest_path, "r", newline="") as f:
        reader = csv.reader(f, delimiter='\t')
        for i, row in tqdm(enumerate(reader)):
            old_rows.append(row)

    manifest_path = os.path.join(out_dir, "manifests")
    Path(manifest_path).mkdir(parents=True, exist_ok=True)
    new_manifest_path = os.path.join(manifest_path, "manifest.tsv")
    log.info("Writing manifest")
    with open(new_manifest_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter='\t')
        for i,row in tqdm(enumerate(old_rows)):
            if i==0:
                writer.writerow(row)
            else:
                path, size = row
                for s in cfg.data_prep.subjects_to_remove:
                    if s not in path:
                        writer.writerow(row)

@hydra.main(version_base=None, config_path="../conf")
def main(cfg: DictConfig) -> None:
    data_cfg = cfg.data
    dataset = build_dataset(data_cfg, preprocessor_cfg=cfg.preprocessor)
    
    assert hasattr(dataset, "files")

    files = dataset.files
    #random.shuffle(files)
    #files = files[:]
    root_dir = dataset.root_dir
    write_manifest(root_dir, cfg.data_prep.out_dir, cfg)

if __name__=="__main__":
    main()

