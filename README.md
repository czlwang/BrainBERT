# BrainBERT

BrainBERT is an modeling approach for learning self-supervised representations of intracranial electrode data. See [paper](https://arxiv.org/abs/2302.14367) for details.

We provide the training pipeline below.

The trained weights have been released (see below) and pre-training data is available upon request.

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
