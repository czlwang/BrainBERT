from .base_criterion import BaseCriterion
import torch
from torch import nn
from criterions import register_criterion

@register_criterion("pretrain_masked_criterion")
class PretrainMaskedCriterion(BaseCriterion):
    def __init__(self):
        super(PretrainMaskedCriterion, self).__init__()
        pass

    def build_criterion(self, cfg):
        self.cfg = cfg

    def forward(self, model, batch, device):
        #x = batch[:,:10] #potentially don't move to device if dataparallel
        pad_mask = batch["attn_mask"].to(device)
        masked_input = batch["masked_input"].to(device) #potentially don't move to device if dataparallel
        mask = batch["mask_label"].bool().to(device)
        output, pos_enc = model.forward(masked_input, pad_mask)
        labels = batch["target"].to(device)
        true_activity = labels.masked_select(mask)
        predicted = output.masked_select(mask)
        l1 = torch.mean(torch.abs(true_activity - predicted))
        non_zero_idxs = torch.abs(true_activity)>1
        non_zero = torch.mean(torch.abs(true_activity[non_zero_idxs] - predicted[non_zero_idxs]))
        content_aware_loss = self.cfg.alpha*non_zero
        loss = l1 + content_aware_loss
        output_log_spec = output[1].detach().cpu()
        content_l1 = non_zero
        wav = batch["wavs"][1]
        images = {"input_spectrogram": masked_input[1].detach().cpu(),
                  "ground_truth": labels[1].detach().cpu(),
                  "pred_spectrogram": output_log_spec,
                  "pos_enc": pos_enc[0].detach().cpu(),
                  "wav": wav}
        logging_output = {"loss": loss.item(), 
                          "images": images,
                          "l1_loss": l1.item(),
                          "content_l1": content_l1.item(),
                          "content_aware_loss": content_aware_loss.item()}
        return loss, logging_output

