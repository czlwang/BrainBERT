#usage
#python3 -m testing.run_fewshot_training_tests +exp=finetune ++exp.runner.num_workers=0 +data=onset_finetuning +model=deep_linear_wav_baseline +task=baseline_wav_task +criterion=baseline_criterion ++exp.runner.scheduler.name=reduce_on_plateau ++exp.runner.log_step=100 +preprocessor=wav_preprocessor ++data.electrodes=["T1cIe11"] ++exp.runner.total_steps=10 ++data.delta=-2.5 ++data.duration=5.0 ++data.cached_data_array=/storage/czw/self_supervised_seeg/cached_data_arrays ++data.name="onset_finetuning" ++data.train_fewshot=??? ++data.reload_caches=???
from omegaconf import DictConfig, OmegaConf
import hydra
import logging
import os
from data.electrode_selection import get_clean_laplacian_electrodes
import json
from .utils import run_electrode_test
import numpy as np
from pathlib import Path

log = logging.getLogger(__name__)

@hydra.main(config_path="../conf")
def main(cfg: DictConfig) -> None:
    log.info(f"Run testing for all training_data_percentages in one electrodes in one test_subject")
    log.info(OmegaConf.to_yaml(cfg, resolve=True))
    log.info(f'Working directory {os.getcwd()}')
    
    out_dir = cfg.test.out_dir
    log.info(f'Out directory {out_dir}')
    Path(out_dir).mkdir(exist_ok=True, parents=True)

    all_test_results = {}
    train_num = range(cfg.test.test_ex_min,cfg.test.test_ex_max,cfg.test.test_ex_step)
    cfg.data.reload_caches=False
    for train_n in train_num:
        cfg.data.train_fewshot = train_n
        test_result = run_electrode_test(cfg)
        cfg.data.reload_caches=False #don't need to cache after first run
        all_test_results[train_n] = test_result

    out_json = os.path.join(out_dir, "all_test_results.json")
    with open(out_json, "w") as f:
        json.dump(all_test_results, f)
    log.info(f"Wrote test results to {out_json}")

if __name__ == "__main__":
    main()
