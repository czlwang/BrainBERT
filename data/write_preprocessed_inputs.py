#to write features to disk
#python3 -m data.write_preprocessed_inputs +data=tf_unmasked +data_prep=cwt_to_disk ++data.data=all_day_data
from multiprocessing import Process
from omegaconf import DictConfig, OmegaConf
import hydra
import models
import tasks
from pretrain.runner import Runner
from datasets import build_dataset
import logging
import os
from pathlib import Path
import numpy as np
from tqdm import tqdm as tqdm
import csv
import yaml

#take dataset with wavs and write it to cache
log = logging.getLogger(__name__)

def write_features(pid, root_dir, files, extracter, cfg):
    absolute_root = Path(cfg.data_prep.out_dir).resolve()

    dirs_to_create = set([os.path.dirname(x) for x in files])
    for out_dir in dirs_to_create:
        cached_absolute_path = os.path.join(absolute_root, out_dir)
        Path(cached_absolute_path).mkdir(parents=True, exist_ok=True)

    for raw_file_path in tqdm(files):
        cached_absolute_path = os.path.join(absolute_root, raw_file_path)
        absolute_path = os.path.join(root_dir, raw_file_path)
        inputs = np.load(absolute_path)
        feature = extracter(inputs)
        np.save(cached_absolute_path, feature)

def write_manifest(files, cfg):
    #map the original file to the cached feature
    absolute_root = Path(cfg.data_prep.out_dir).resolve()
    paths = []
    for raw_file_path in tqdm(files):
        paths.append((raw_file_path, raw_file_path))

    manifest_path = os.path.join(absolute_root, "manifests")
    Path(manifest_path).mkdir(parents=True, exist_ok=True)
    cfg_path = os.path.join(manifest_path, "config.yaml")
    manifest_path = os.path.join(manifest_path, "manifest.tsv")
    log.info("Writing cfg")
    with open(cfg_path, 'w') as yamlfile:
        OmegaConf.save(config=cfg, f=yamlfile.name)

    log.info("Writing manifest")
    with open(manifest_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow((absolute_root,))
        writer.writerow(('orig_file', 'cached_feature'))
        for row in tqdm(paths):
            writer.writerow(row)

@hydra.main(version_base=None, config_path="../conf")
def main(cfg: DictConfig) -> None:
    log.info("Writing data to disk")
    data_cfg = cfg.data
    dataset = build_dataset(data_cfg, preprocessor_cfg=cfg.preprocessor)
    
    assert hasattr(dataset, "files")
    assert hasattr(dataset, "extracter")
    assert "preprocessor" in data_cfg

    extracter = dataset.extracter
    files = dataset.files
    root_dir = dataset.root_dir

    ps = []
    n=cfg.data_prep.n_workers
    step = int(len(files)/n) + 1
    ranges = [(i*step, (i+1)*step) for i in range(n)]
    for index in range(n):
        start, end = ranges[index]
        idx_slice = files[start:end]
        log.info(f'Main    : create and start process {index} with {start} to {end}')
        #write_features(root_dir, files, extracter, cfg)
        pid = index
        x = Process(target=write_features, args=(pid, root_dir, idx_slice, extracter, cfg))
        ps.append(x)
        x.start()

    for index, process in enumerate(ps):
        log.info(f'Main    : before joining process {index}')
        process.join()
        log.info("Main    : process %d done", index)

    write_manifest(files, cfg)

if __name__=="__main__":
    main()
