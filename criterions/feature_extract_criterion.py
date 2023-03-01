import torch
from .base_criterion import BaseCriterion
from torch import nn
from criterions import register_criterion

@register_criterion("feature_extract_criterion")
class FeatureExtractCriterion(BaseCriterion):
    def __init__(self):
        super(FeatureExtractCriterion, self).__init__()
        pass

    def build_criterion(self, cfg):
        self.cfg = cfg
        self.sigmoid = nn.Sigmoid()
        self.loss_fn = nn.BCEWithLogitsLoss(reduction="mean")

    def forward(self, model, batch, device, return_predicts=False):
        #TODO fix the dataset here. 
        inputs = batch["input"].to(device) #potentially don't move to device if dataparallel
        output = model.forward(inputs)
        labels = torch.FloatTensor(batch["labels"]).to(output.device)
        output = output.squeeze(-1)
        loss = self.loss_fn(output, labels)
        images = {"wav": batch["wavs"][0],
                  "wav_label": batch["labels"][0]}
        if return_predicts:
            predicts = self.sigmoid(output).squeeze().detach().cpu().numpy()
            logging_output = {"loss": loss.item(),
                              "predicts": predicts,
                              "images": images}
        else:
            logging_output = {"loss": loss.item(),
                              "images": images}
        return loss, logging_output

