from multiprocessing import Process
from omegaconf import DictConfig, OmegaConf
import hydra
import logging
from pathlib import Path
import os
import json
from datasets import build_dataset
from .electrode_selection import get_clean_laplacian_electrodes
from .utils import write_manifest
import csv
import glob
import time
from tqdm import tqdm as tqdm
import numpy as np

log = logging.getLogger(__name__)

def write_manifest(manifest_path, root_out, paths, lengths):
    absolute_path = Path(root_out).resolve()
    manifest_path = os.path.join(manifest_path, "manifests")
    Path(manifest_path).mkdir(exist_ok=True, parents=True)
    manifest_path = os.path.join(manifest_path, "manifest.tsv")
    with open(manifest_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow((str(absolute_path),))
        for row in zip(paths, lengths):
            writer.writerow(row)

def write_trial_data(root_out, trial_id, data_cfg):
    log.info(f'Writing {trial_id}')
    subject_id = data_cfg.subject
    data_cfg.brain_runs = [trial_id]
    electrodes = data_cfg.electrodes
    paths, lengths = [], []
    trial_absolute_path = os.path.join(root_out, subject_id, trial_id)
    Path(trial_absolute_path).mkdir(exist_ok=True, parents=True)

    global_i = 0
    for electrode in tqdm(electrodes):#iterate over electrodes here to save memory
        data_cfg_copy = data_cfg.copy()
        data_cfg_copy.electrodes = [electrode]
        dataset = build_dataset(data_cfg_copy)
        for i in range(len(dataset)):
            print("index", global_i)
            example = dataset[i]["input"].squeeze()
            file_name = f'{global_i}.npy'
            relative_path = os.path.join(subject_id, trial_id, file_name)
            save_path = os.path.join(trial_absolute_path, file_name)
            np.save(save_path, example)
            paths.append(str(relative_path))
            lengths.append(example.shape[0])
            global_i += 1
    manifest_path = os.path.join(root_out, "manifests", subject_id, trial_id)
    Path(manifest_path).mkdir(exist_ok=True, parents=True)
    write_manifest(manifest_path, root_out, paths, lengths)
    return paths, lengths

def write_absolute_manifests(root_out):
    absolute_path = Path(root_out).resolve()
    all_tsvs = glob.glob(os.path.join(absolute_path, "manifests/*/*/*/*.tsv"))
    header = ""
    all_rows = []
    for f in all_tsvs:
        with open(f, "r") as fd: 
            rd = csv.reader(fd, delimiter="\t", quotechar='"')
            for i, row in enumerate(rd):
                if i==0:
                    header = row
                else:
                    all_rows.append(row)

    manifest_path = os.path.join(root_out, "manifests", "manifest.tsv")
    with open(manifest_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(header)
        for row in all_rows:
            writer.writerow(row)

def single_process(cfg, subject_splits):
    root_out = cfg.data_prep.out_dir
    data_cfg = cfg.data

    paths, lengths = [], []
    Path(root_out).mkdir(exist_ok=True, parents=True)

    for subject in subject_splits:
        print("subject", subject)
        for trial in subject_splits[subject]:
            print("trial", trial)
            data_cfg.brain_runs=[trial]
            data_cfg.electrodes = get_clean_laplacian_electrodes(subject, data_root=cfg.data.raw_brain_data_dir)
            data_cfg.subject = subject
            write_trial_data(root_out, trial, data_cfg)

@hydra.main(version_base=None, config_path="../conf")
def main(cfg: DictConfig) -> None:
    log.info("Writing data to disk")
    log.info(OmegaConf.to_yaml(cfg, resolve=True))
    log.info(f'Working directory {os.getcwd()}')

    start_time = time.time()
    pretrain_split_path = cfg.data_prep.pretrain_split
    with open(pretrain_split_path) as f:
        pretrain_split = json.load(f)

    subject_splits = {}
    for i,k in enumerate(pretrain_split):
        idx = i%2
        if idx not in subject_splits:
            subject_splits[idx] = {}
        subject_splits[idx][k] = pretrain_split[k]

    #subject splits maps process_id to a subset of pretrain split
    ps = []
    for i in subject_splits:
        x = Process(target=single_process, args=(cfg, subject_splits[i]))
        ps.append(x)
        x.start()

    for index, process in enumerate(ps):
        log.info(f'Main    : before joining process {index}')
        process.join()
        log.info("Main    : process %d done", index)

    root_out = cfg.data_prep.out_dir
    write_absolute_manifests(root_out)
    end_time = time.time()
    log.info(f'total time {(end_time - start_time)/60} minutes')


if __name__=="__main__":
    main()
