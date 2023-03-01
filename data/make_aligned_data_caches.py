from omegaconf import DictConfig, OmegaConf
import hydra
import models
import tasks
import logging
import os
from data.electrode_selection import get_clean_laplacian_electrodes
from data.subject_data import SubjectData
from data.speech_nonspeech_subject_data import NonLinguisticSubjectData, SentenceOnsetSubjectData
import json

log = logging.getLogger(__name__)

def create_subj_cache(data_cfg, brain_runs, electrodes, cfg):
    cache_path = None
    if "cache_input_features" in data_cfg:
        cache_path = data_cfg.cache_input_features

    subject_test_results = {}
    data_cfg.electrodes = electrodes
    data_cfg.brain_runs = brain_runs
    if cache_path is not None:
        #cache_path needs to identify the pretrained model
        e_cache_path = os.path.join(cache_path, data_cfg.subject, data_cfg.name ,e)
        log.info(f"logging input features in {e_cache_path}")
        data_cfg.cache_input_features = e_cache_path
    subj_data = SentenceOnsetSubjectData(data_cfg)

@hydra.main(config_path="../conf")
def main(cfg: DictConfig) -> None:
    log.info(f"Run testing for all electrodes in all test_subjects")
    log.info(OmegaConf.to_yaml(cfg, resolve=True))
    log.info(f'Working directory {os.getcwd()}')

    test_split_path = cfg.exp.test_split_path 
    with open(test_split_path, "r") as f:
        test_splits = json.load(f)

    data_cfg = cfg.data
    all_test_results = {}
    for subj in test_splits:
        log.info(f"Subject {subj}")
        data_cfg.subject = subj
        electrodes = get_clean_laplacian_electrodes(subj)
        create_subj_cache(data_cfg, test_splits[subj], electrodes, cfg)

if __name__ == "__main__":
    main()

