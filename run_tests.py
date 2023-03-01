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

def run_subject_test(data_cfg, brain_runs, electrodes, cfg):
    data_cfg_copy = data_cfg.copy()
    cache_path = None
    if "cache_input_features" in data_cfg_copy:
        cache_path = data_cfg_copy.cache_input_features

    subject_test_results = {}
    for e in electrodes:
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
        model = task.build_model(cfg.model)
        criterion = task.build_criterion(cfg.criterion)
        runner = Runner(cfg.exp.runner, task, model, criterion)
        best_model = runner.train()
        test_results = runner.test(best_model)
        subject_test_results[e] = test_results
    return subject_test_results

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

@hydra.main(config_path="conf")
def main(cfg: DictConfig) -> None:
    log.info(f"Run testing for all electrodes in all test_subjects")
    log.info(OmegaConf.to_yaml(cfg, resolve=True))
    out_dir = os.getcwd()
    log.info(f'Working directory {os.getcwd()}')
    if "out_dir" in cfg.test:
        out_dir = cfg.test.out_dir
    log.info(f'Output directory {out_dir}')

    test_split_path = cfg.test.test_split_path 
    with open(test_split_path, "r") as f:
        test_splits = json.load(f)

    test_electrodes = None #For the topk. Omit this argument if you want everything
    if "test_electrodes_path" in cfg.test and cfg.test.test_electrodes_path != "None": #very hacky
        test_electrodes_path = cfg.test.test_electrodes_path 
        test_electrodes_path = os.path.join(test_electrodes_path, cfg.data.name)
        test_electrodes_path = os.path.join(test_electrodes_path, "linear_results.json")
        with open(test_electrodes_path, "r") as f:
            test_electrodes = json.load(f)

    data_cfg = cfg.data
    all_test_results = {}
    for subj in test_splits:
        subj_test_results = {}
        Path(out_dir).mkdir(exist_ok=True, parents=True)
        out_path = os.path.join(out_dir, "all_test_results")
        log.info(f"Subject {subj}")
        data_cfg.subject = subj
        if test_electrodes is not None:
            if subj not in test_electrodes:
                continue
            electrodes = test_electrodes[subj]
        else:
            electrodes = get_clean_laplacian_electrodes(subj)
        subject_test_results = run_subject_test(data_cfg, test_splits[subj], electrodes, cfg)
        all_test_results[subj] = subject_test_results
        subj_test_results[subj] = subject_test_results

        out_json_path = os.path.join(out_path, subj)
        Path(out_json_path).mkdir(exist_ok=True, parents=True)
        out_json = os.path.join(out_json_path, "subj_test_results.json")
        with open(out_json, "w") as f:
            json.dump(subj_test_results, f)
        log.info(f"Wrote test results to {out_json}")
    write_summary(all_test_results, out_path)

if __name__ == "__main__":
    main()
