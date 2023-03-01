from torch.nn.utils.rnn import pad_sequence
import torch

def make_pad_mask(batched_input, lengths):
    pad_mask = torch.ones(batched_input.shape[:-1]) #[batch, len]

    for i in range(pad_mask.shape[0]):
        pad_mask[i,lengths[i]:] = 0

    pad_mask = ~pad_mask.bool() 
    return pad_mask

def spec_collator(batch):
    input_specs = [b["masked_input"] for b in batch]
    mask_labels = [b["mask_label"] for b in batch]
    targets = [b["target"] for b in batch]
    lengths = [b["length"] for b in batch]
    wavs = [b["wav"] for b in batch]

    batched_input = pad_sequence(input_specs, batch_first=True)
    batched_target = pad_sequence(targets, batch_first=True)
    batched_mask_label = pad_sequence(mask_labels, batch_first=True)

    attn_mask = make_pad_mask(batched_input, lengths)

    batch = {"attn_mask": attn_mask,
             "masked_input": batched_input,
             "target": batched_target,
             "mask_label": batched_mask_label,
             "wavs": wavs}
    return batch

def wav_collator(batch):
    wavs = [torch.Tensor(b["input"]).unsqueeze(0) for b in batch]
    wavs = pad_sequence(wavs, batch_first=True)
    return {"input":wavs,
           }

def baseline_wav_collator(batch):
    labels = [b["label"] for b in batch]
    wavs = [torch.Tensor(b["input"]) for b in batch]
    wavs = pad_sequence(wavs, batch_first=True)

    lengths = [b["length"] for b in batch]

    return {"input":wavs,
            "labels":labels,
           }

def finetune_collator(batch):
    specs = [b["input"] for b in batch]
    specs = pad_sequence(specs, batch_first=True)
    labels = [b["label"] for b in batch]
    wavs = [b["wav"] for b in batch]

    lengths = [b["length"] for b in batch]
    pad_mask = make_pad_mask(specs, lengths)

    return {"input":specs,
            "labels":labels,
            "wavs": wavs,
            "pad_mask": pad_mask}

def feature_extracter_collator(batch):
    specs = [b["input"] for b in batch]
    specs = pad_sequence(specs, batch_first=True)
    labels = [b["label"] for b in batch]
    wavs = [b["wav"] for b in batch]

    lengths = [b["length"] for b in batch]

    return {"input":specs,
            "labels":labels,
            "wavs": wavs}
