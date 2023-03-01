#to write features to disk
# python3 -m data.throw_out_zeros +data=pretrain_wavs_from_disk ++data.data=pretrain_data/manifests +preprocessor=wav_preprocessor +data_prep=throw_out_zeros
from multiprocessing import Process, Queue
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
import json

#take dataset with wavs and write it to cache
log = logging.getLogger(__name__)

def check_features(pid, root_dir, files, cfg, qq):
    bad_files = []
    for raw_file_path in tqdm(files):
        absolute_path = os.path.join(root_dir, raw_file_path)
        inputs = np.load(absolute_path)
        #print(inputs.shape)
        inputs = inputs.astype('float32')
        if (inputs[0]==inputs).all():
            print("BAD FILE", raw_file_path)
            bad_files.append(raw_file_path)
    bad_files_q = qq.get()
    bad_files_q += bad_files
    qq.put(bad_files_q)

def write_manifest(root_dir, all_bad_files):
    #map the original file to the cached feature
    manifest_path = os.path.join(root_dir, "manifests")
    Path(manifest_path).mkdir(parents=True, exist_ok=True)
    old_manifest_path = os.path.join(manifest_path, "manifest.tsv")
    old_rows = []
    with open(old_manifest_path, "r", newline="") as f:
        reader = csv.reader(f, delimiter='\t')
        for i, row in tqdm(enumerate(reader)):
            old_rows.append(row)

    new_manifest_path = os.path.join(manifest_path, "new_manifest.tsv")
    log.info("Writing manifest")
    with open(new_manifest_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter='\t')
        for i,row in tqdm(enumerate(old_rows)):
            if i==0:
                writer.writerow(row)
            else:
                path, size = row
                if path not in all_bad_files:
                    writer.writerow(row)

@hydra.main(version_base=None, config_path="../conf")
def main(cfg: DictConfig) -> None:
    log.info("Writing data to disk")
    data_cfg = cfg.data
    dataset = build_dataset(data_cfg, preprocessor_cfg=cfg.preprocessor)
    
    assert hasattr(dataset, "files")

    files = dataset.files
    random.shuffle(files)
    files = files[:]
    root_dir = dataset.root_dir

    ps = []
    n=cfg.data_prep.n_workers
    step = int(len(files)/n) + 1
    ranges = [(i*step, (i+1)*step) for i in range(n)]
    qq = Queue()
    qq.put([])
    for index in range(n):
        start, end = ranges[index]
        idx_slice = files[start:end]
        log.info(f'Main    : create and start process {index} with {start} to {end}')
        pid = index
        x = Process(target=check_features, args=(pid, root_dir, idx_slice, cfg, qq))
        ps.append(x)
        x.start()

    for index, process in enumerate(ps):
        log.info(f'Main    : before joining process {index}')
        process.join()
        log.info("Main    : process %d done", index)
    all_bad_files = qq.get()

    out_path = "/storage/czw/self_supervised_seeg/data/all_zero_files_1.json"
    with open(out_path, "w") as f:
        json.dump(all_bad_files, f)
    write_manifest(root_dir, all_bad_files)

if __name__=="__main__":
    main()

