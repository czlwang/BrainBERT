# reads the ROC-AUC of linear results and stores the top k
#usage
#python3 -m testing.process_linear_results +test=process_linear_results
from omegaconf import DictConfig, OmegaConf
import hydra
import logging
import os
from data.electrode_selection import get_clean_laplacian_electrodes
import json
from .utils import run_electrode_test
import numpy as np
import glob
from pathlib import Path

log = logging.getLogger(__name__)

@hydra.main(config_path="../conf", version_base=None)
def main(cfg: DictConfig) -> None:
    log.info(f"Processing linear results")
    log.info(OmegaConf.to_yaml(cfg, resolve=True))
    log.info(f'Working directory {os.getcwd()}')
    results_path = cfg.test.linear_results_path
    results_files = glob.glob(os.path.join(results_path, "*", "*"))
    all_results = []
    for path in results_files:
        subj = path.split("/")[-2]
        with open(path, "r") as f:
            subj_res = json.load(f)
            subj_res = subj_res[subj]
            for e in subj_res.keys():
                name = f'{subj}_{e}'
                roc_auc = subj_res[e]['roc_auc']
                if "single_subject" not in cfg.test or cfg.test.single_subject==subj:
                    all_results.append((name, roc_auc))
    k = cfg.test.topk
    topk = sorted(all_results, key=lambda x: x[1])[-k:]
    topk_results = {}
    for k, _ in topk:
        subj, e = k.split("_")
        if subj not in topk_results:
            topk_results[subj] = []
        topk_results[subj].append(e)
    Path(cfg.test.out_dir).mkdir(exist_ok=True, parents=True)
    out_path = os.path.join(cfg.test.out_dir, "linear_results.json")
    with open(out_path, "w") as f:
        json.dump(topk_results, f)
main()
