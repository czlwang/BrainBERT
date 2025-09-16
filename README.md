# BrainBERT

BrainBERT is an modeling approach for learning self-supervised representations of intracranial electrode data. See [paper](https://arxiv.org/abs/2302.14367) for details.

We provide the training pipeline below.

The trained weights have been released (see below) and pre-training data can be found at [braintreebank.dev](https://braintreebank.dev)

## Installation
Requirements:
- pytorch >= 1.12.1
- [pytorch gradual warmup scheduler](https://github.com/ildoonet/pytorch-gradual-warmup-lr)

```
pip install -r requirements.txt
```

### Input
It is expected that the input is intracranial electrode data that has been Laplacian re-referenced.

## Using BrainBERT embeddings
- pretrained weights are available [here](https://drive.google.com/file/d/14ZBOafR7RJ4A6TsurOXjFVMXiVH6Kd_Q/view?usp=sharing)
- see `notebooks/demo.ipynb` for an example input and example embedding

## Upstream
### BrainBERT pre-training data
The data directory should be structured as:
```
/pretrain_data
  |_manifests
    |_manifests.tsv  <-- each line contains the path to the example and the length
  |_<subject>
    |_<trial>
      |_<example>.npy
```
If using the data from the Brain Treebank, the data can be written using this command:
```
python3 -m data.write_pretrain_data_wavs +data=pretraining_template.yaml \
+data_prep=write_pretrain_split ++data.duration=5 \
++data_prep.pretrain_split=data/pretrain_split_trials.json \
++data_prep.out_dir=pretrain_data \
++data.raw_brain_data_dir=/path/to/braintreebank_data/
```
This command expects the Brain Treebank data to have the following structure:
```
/braintreebank_data
  |_electrode_labels
  |_subject_metadata
  |_localization
  |_all_subject_data
    |_sub_*_trial*.h5
```

### BrainBERT pre-training
```
python3 run_train.py +exp=spec2vec ++exp.runner.device=cuda ++exp.runner.multi_gpu=True \
  ++exp.runner.num_workers=64 +data=masked_spec +model=masked_tf_model_large \
  +data.data=/path/to/data ++data.val_split=0.01 +task=fixed_mask_pretrain.yaml \
  +criterion=pretrain_masked_criterion +preprocessor=stft ++data.test_split=0.01 \
  ++task.freq_mask_p=0.05 ++task.time_mask_p=0.05 ++exp.runner.total_steps=500000
```
Example parameters:
```
/path/to/data = /storage/user123/self_supervised_seeg/pretrain_data/manifests
```
