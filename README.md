# BrainBERT

BrainBERT is an modeling approach for learning self-supervised representations of intracranial electrode data.

We provide the training and fine-tuning pipeline instructions below.

The data will be released as well. (Check back here for updates).

### Input
It is expected that the input is intracranial electrode data that has been Laplacian re-referenced.

## Upstream
### BrainBERT pre-training
```
python3 run_train.py +exp=spec2vec ++exp.runner.device=cuda ++exp.runner.multi_gpu=True ++exp.runner.num_workers=64 +data=masked_spec +model=masked_tf_model_large +data.data=/path/to/data ++data.val_split=0.01 +task=fixed_mask_pretrain.yaml +criterion=pretrain_masked_criterion +preprocessor=stft ++data.test_split=0.01 ++task.freq_mask_p=0.05 ++task.time_mask_p=0.05 ++exp.runner.total_steps=500000
```
Example parameters:
```
/path/to/data = /storage/user123/self_supervised_seeg/pretrain_data/manifests
```

## Downstream
### Get BrainBERT test performance with finetuning on the sentence onset task
```
python3 run_tests.py +exp=finetune +data=finetuning_template +model=finetune_model +task=finetune_task +criterion=finetune_criterion +preprocessor=superlet ++model.upstream_ckpt=/path/to/checkpoint ++data.reload_caches=True ++data.name=onset_finetuning +test=held_out_subjects
```
Example parameters:
```
/path/to/checkpoint = /storage/user123/self_supervised_seeg/pretrained_weights/superlet_large_pretrained.pth 
```

### Get BrainBERT test performance with no-finetuning on the sentence onset task
```
python3 run_tests.py +exp=feature_extract +data=finetuning_template +model=feature_extract_model +task=feature_extract +criterion=feature_extract_criterion +preprocessor=superlet_pretrained ++preprocessor.upstream_ckpt=/path/to/upstream ++data.reload_caches=True ++data.name=onset_finetuning +test=held_out_subjects 
```
Example parameters:
```
/path/to/upstream = /storage/user123/self_supervised_seeg/pretrained_weights/superlet_large_pretrained.pth
```
