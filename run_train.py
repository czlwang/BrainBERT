#example
#python3 run_train.py +exp=spec2vec ++exp.runner.device=cuda ++exp.runner.multi_gpu=True ++exp.runner.num_workers=16 +data=masked_spec +model=debug_model +data.data=/storage/czw/self_supervised_seeg/all_electrode_data/manifests
from omegaconf import DictConfig, OmegaConf
import hydra
import models
import tasks
from runner import Runner
import logging
import os

log = logging.getLogger(__name__)

@hydra.main(config_path="conf")
def main(cfg: DictConfig) -> None:
    log.info("Training")
    log.info(OmegaConf.to_yaml(cfg, resolve=True))
    log.info(f'Working directory {os.getcwd()}')
    task = tasks.setup_task(cfg.task)
    task.load_datasets(cfg.data, cfg.preprocessor)
    model = task.build_model(cfg.model)
    criterion = task.build_criterion(cfg.criterion)
    runner = Runner(cfg.exp.runner, task, model, criterion)
    best_model = runner.train()
    runner.test(best_model)

if __name__ == "__main__":
    main()
