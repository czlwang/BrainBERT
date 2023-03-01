#usage
#python3 run_train.py +exp=spec2vec ++exp.runner.device=cuda ++exp.runner.multi_gpu=False ++exp.runner.num_workers=0 +data=timestamp_data +model=debug_finetune_model ++exp.task.name=debug_finetune_task ++exp.criterion.name=debug_finetune_criterion ++exp.runner.total_steps=1000 ++model.frozen_upstream=True ++exp.runner.checkpoint_step=-1
import logging
import numpy as np
import models
from torch.utils import data
import torch
from tasks import register_task
from tasks.base_task import BaseTask
from tasks.batch_utils import baseline_wav_collator
from util.tensorboard_utils import plot_tensorboard_line
from sklearn.metrics import roc_auc_score, f1_score

log = logging.getLogger(__name__)

@register_task(name="baseline_wav_task")
class BaselineWavTask(BaseTask):
    def __init__(self, cfg):
        super(BaselineWavTask, self).__init__(cfg)

    def build_model(self, cfg):
        #assert hasattr(self, "dataset")
        input_dim = self.dataset.get_input_dim()
        return models.build_model(cfg, input_dim)
        
    @classmethod
    def setup_task(cls, cfg):
        return cls(cfg)

    def get_valid_outs(self, model, valid_loader, criterion, device):
        model.eval()
        all_outs = {"loss":0}
        predicts, labels = [], []
        with torch.no_grad():
            for batch in valid_loader:
                batch["input"] = batch["input"].to(device)
                _, valid_outs = criterion(model, batch, device, return_predicts=True)

                predicts.append(valid_outs["predicts"])
                labels.append(batch["labels"])
                all_outs["loss"] += valid_outs["loss"]
        labels = np.array([x for y in labels for x in y])
        predicts = [np.array([p]) if len(p.shape)==0 else p for p in predicts]
        predicts = np.concatenate(predicts)
        roc_auc = roc_auc_score(labels, predicts)
        all_outs["loss"] /= len(valid_loader)
        all_outs["roc_auc"] = roc_auc
        f1 = f1_score(labels, np.round(predicts))
        all_outs["f1"] = f1
        return all_outs

    def get_batch_iterator(self, dataset, batch_size, shuffle=True, **kwargs):
        return data.DataLoader(dataset, batch_size=batch_size, collate_fn=baseline_wav_collator, **kwargs)

    def output_logs(self, train_logging_outs, val_logging_outs, writer, global_step):
        val_auc_roc = val_logging_outs["roc_auc"]
        val_f1 = val_logging_outs["f1"]
        if writer is not None:
            writer.add_scalar("valid_roc_auc", val_auc_roc, global_step)
            writer.add_scalar("valid_f1", val_f1, global_step)
        log.info(f'valid_roc_auc: {val_auc_roc}')

        image = train_logging_outs["images"]["wav"]
        label = train_logging_outs["images"]["wav_label"]
        tb_image = plot_tensorboard_line(image, title=label)
        if writer is not None:
            writer.add_image("raw_wave", tb_image, global_step)

