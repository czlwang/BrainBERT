import numpy as np
from omegaconf import DictConfig, OmegaConf
import hydra
import models
import tasks
from runner import Runner
import logging
import os
from data.electrode_selection import get_clean_laplacian_electrodes
import json
from pathlib import Path

log = logging.getLogger(__name__)

def get_electrode_data(data_cfg, brain_runs, electrodes, cfg):
    data_cfg_copy = data_cfg.copy()
    cache_path = None
    if "cache_input_features" in data_cfg_copy:
        cache_path = data_cfg_copy.cache_input_features

    subject_test_results = {}
    positive_count, negative_count = 0,0
    e = electrodes[0]
    data_cfg_copy.electrodes = [e]
    data_cfg_copy.brain_runs = brain_runs
    if cache_path is not None:
        #cache_path needs to identify the pretrained model
        e_cache_path = os.path.join(cache_path, data_cfg_copy.subject, data_cfg_copy.name ,e)
        log.info(f"logging input features in {e_cache_path}")
        data_cfg_copy.cache_input_features = e_cache_path
    cfg.data = data_cfg_copy
    task = tasks.setup_task(cfg.task)
    task.load_datasets(cfg.data, cfg.preprocessor)
    labels = np.array([e["label"] for e in task.dataset])
    positive_count += sum(labels==1)
    negative_count += sum(labels==0)
    return positive_count, negative_count

def write_summary(all_test_results, out_path):
    out_json = os.path.join(out_path, "all_test_results.json")
    with open(out_json, "w") as f:
        json.dump(all_test_results, f)

    out_json = os.path.join(out_path, "summary.json")
    all_rocs = []
    for s in all_test_results:
        for e in all_test_results[s]:
            all_rocs.append(all_test_results[s][e]["roc_auc"])

    summary_results = {"avg_roc_auc": np.mean(all_rocs), "std_roc_auc": np.std(all_rocs)}
    with open(out_json, "w") as f:
        json.dump(summary_results, f)

    log.info(f"Wrote test results to {out_path}")

def get_dataset_stats(data_cfg, test_splits, cfg):
    positive_count, negative_count = [], []
    subj_results = {}
    for subj in test_splits:
        subj_test_results = {}
        log.info(f"Subject {subj}")
        data_cfg.subject = subj
        electrodes = get_clean_laplacian_electrodes(subj)
        p, n = get_electrode_data(data_cfg, test_splits[subj], electrodes, cfg)
        positive_count.append(p)
        negative_count.append(n)
        subj_results[subj] = {"positive": p.item(), "negative": n.item()}
    return np.array(positive_count).mean(), np.array(negative_count).mean(), subj_results

@hydra.main(config_path="../conf")
def main(cfg: DictConfig) -> None:
    log.info(f"Get data stats")
    log.info(OmegaConf.to_yaml(cfg, resolve=True))
    out_dir = os.getcwd()
    log.info(f'Working directory {os.getcwd()}')
    if "out_dir" in cfg.test:
        out_dir = cfg.test.out_dir
    log.info(f'Output directory {out_dir}')

    test_split_path = cfg.test.test_split_path 
    with open(test_split_path, "r") as f:
        test_splits = json.load(f)

    data_cfg = cfg.data
    all_test_results = {}
    features = ["onset_finetuning", "speech_finetuning", "rms_finetuning", "pitch_finetuning"]
    for feature in features:
        print("feature", feature)
        data_cfg.name = feature
        p, n, s = get_dataset_stats(data_cfg, test_splits, cfg)
        print("positive", p, "negative", n)
        #all_test_results[feature] = {"positive":p, "negative":n}
        all_test_results[feature] = s
        
    Path(out_dir).mkdir(exist_ok=True, parents=True)
    with open(os.path.join(out_dir, "all_results.json"), "w") as f:
        json.dump(all_test_results, f)

if __name__ == "__main__":
    main()
