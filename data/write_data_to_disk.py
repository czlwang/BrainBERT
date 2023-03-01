#usage: 
#to write wavs to disk
#python3 -m data.write_data_to_disk +data=pretraining.yaml +data_prep=wavs_to_disk ++data.prep.out_dir=all_day_data
#python3 -m data.write_preprocessed_inputs +data=tf_unmasked +data_prep=cwt_to_disk ++data.data=all_day_data
from omegaconf import DictConfig, OmegaConf
import hydra
import logging
from pathlib import Path
import os
from datasets import build_dataset

log = logging.getLogger(__name__)

def write_trial_data(root_out, trial_id, data_cfg):
    subject_id = data_cfg.subject
    data_cfg.brain_runs = [trial_id]
    dataset = build_dataset(data_cfg)
    log.info(f'Writing {trial_id}')
    paths, lengths = [], []
    trial_absolute_path = os.path.join(root_out, subject_id, trial_id)
    Path(trial_absolute_path).mkdir(exist_ok=True, parents=True)
    for i in tqdm(range(len(dataset))):
        example = dataset[i]["input"].squeeze()
        file_name = f'{i}.npy'
        relative_path = os.path.join(subject_id, trial_id, file_name)
        save_path = os.path.join(trial_absolute_path, file_name)
        np.save(save_path, example)
        paths.append(str(relative_path))
        lengths.append(example.shape[0])
    return paths, lengths

@hydra.main(version_base=None, config_path="../conf")
def main(cfg: DictConfig) -> None:
    log.info("Writing data to disk")
    log.info(OmegaConf.to_yaml(cfg, resolve=True))
    log.info(f'Working directory {os.getcwd()}')

    #NOTE: we are going to write the data in chunks. There is a brain_runs argument in data_cfg that we will overwrite in our loop
    root_out = cfg.data_prep.out_dir
    data_cfg = cfg.data

    Path(root_out).mkdir(exist_ok=True, parents=True)
    paths, lengths = [], []
    for trial_id in cfg.data_prep.brain_runs:
        p,l = write_trial_data(root_out, trial_id, data_cfg)
        paths = paths + p
        lengths = lengths + l
    write_manifest(root_out, paths, lengths)

if __name__=="__main__":
    main()
