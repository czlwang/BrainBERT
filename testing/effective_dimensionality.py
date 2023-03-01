from tasks.batch_utils import finetune_collator
import json
from omegaconf import DictConfig, OmegaConf
import hydra
import logging
import os
import torch
import random
import numpy as np
import models
import umap
from data.electrode_selection import get_clean_laplacian_electrodes
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from datasets import build_dataset
from tqdm import tqdm as tqdm
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from torch.utils.data import Subset
from torch.utils import data

log = logging.getLogger(__name__)

def load_model_weights(model, states, multi_gpu):
    if multi_gpu:
        model.module.load_weights(states)
    else:
        model.load_weights(states)

log = logging.getLogger(__name__)

def make_scatter_plot(vecs, labels, dataset, name="scatter"):
    #labels must be numeric
    unique_colors = np.unique(labels)
    colors = np.array(labels)

    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots()
    for color in unique_colors:
        cvecs = vecs[colors==color]
        ax.scatter(cvecs[:,0],
                   cvecs[:,1],
                   color=cmap(color),
                   label=color,
                   s=1)
    ax.legend(markerscale=5)
    plt.savefig(f'{name}.png')

def get_effective_dim(contexts, dataset, args):
    if args.dim_reduce=="pca":
        pca = PCA(n_components=args.n_components)
        reduced = pca.fit_transform(contexts)
    #if args.dim_reduce=="tsne":
    #    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=1000)
    #    reduced = tsne.fit_transform(contexts)
    #if args.dim_reduce=="umap":
    #    reducer = umap.UMAP()
    #    reduced = reducer.fit_transform(contexts)
    ratios = pca.explained_variance_ratio_
    dim = 0
    dist = {i:0 for i in range(len(ratios))}
    while dim < len(ratios):
        percent = np.sum(ratios[:dim])
        if percent > 0.95:
            break
        dist[dim] = percent.item()
        dim += 1
    return dim, dist

def build_model(cfg):
    ckpt_path = cfg.upstream_ckpt
    init_state = torch.load(ckpt_path)
    upstream_cfg = init_state["model_cfg"]
    upstream = models.build_model(upstream_cfg)
    return upstream

def get_embeddings(dataset, model, raw_spec=False):
    embeds, labels = [], []
    if model is not None:
        model.eval()
    all_idxs = list(range(len(dataset)))
    #random.shuffle(all_idxs)
    #for item in tqdm(dataset):
    subset = Subset(dataset, [x for x in range(500)])
    loader = data.DataLoader(subset, batch_size=64, collate_fn=finetune_collator)

    for batch in tqdm(loader):
        if raw_spec:
            out = batch["input"]
        else:
            inputs = batch["input"].to('cuda')
            mask = torch.zeros((inputs.shape[:2])).bool().to('cuda')
            with torch.no_grad():
                out = model.forward(inputs, mask, intermediate_rep=True)
        middle = out.shape[1]
        #embed = out[:,middle-5:middle+5,:].mean(axis=1) #TODO remove
        embed = out.mean(axis=1)
        #embed = out[:,random.randint(0,62),:]
        if np.any(np.array(batch["labels"])==0):
            import pdb; pdb.set_trace()
        embeds.append(embed.cpu().numpy())
    embeds = np.concatenate(embeds)
    return embeds, labels

@hydra.main(config_path="../conf")
def main(cfg: DictConfig) -> None:
    log.info(f"Find effective dimensionality")
    log.info(OmegaConf.to_yaml(cfg, resolve=True))
    log.info(f'Working directory {os.getcwd()}')

    raw_spec = cfg.test.raw_spec
    model = None
    if not raw_spec:
        model = build_model(cfg.test)
        model = torch.nn.DataParallel(model)
        model.to('cuda')
        ckpt_path = cfg.test.upstream_ckpt
        init_state = torch.load(ckpt_path)
        load_model_weights(model, init_state['model'], True)

    log.info(f'Use {torch.cuda.device_count()} GPUs')

    test_split_path = cfg.test.test_split_path 
    with open(test_split_path, "r") as f:
        test_splits = json.load(f)

    all_results = {}
    Path(cfg.test.out_dir).mkdir(parents=True, exist_ok=True)
    for subject in test_splits:
        electrodes = get_clean_laplacian_electrodes(subject)
        all_results[subject] = {}
        subj_data_cfg = cfg.data.copy()
        subj_data_cfg.subject = subject
        random.shuffle(electrodes)
        for e in electrodes:#[:100]:
            data_cfg = subj_data_cfg.copy()
            data_cfg.electrodes=[e]
            dataset = build_dataset(data_cfg, preprocessor_cfg=cfg.preprocessor)
            embeddings, labels = get_embeddings(dataset, model, cfg.test.raw_spec)
            dim, dist = get_effective_dim(embeddings, dataset, cfg.test)
            all_results[subject][e] = dim
            e_out_dir = os.path.join(cfg.test.out_dir, subject, e)
            Path(e_out_dir).mkdir(parents=True, exist_ok=True)
            print(dim)
            with open(os.path.join(e_out_dir, "dim_results.json"), "w") as f:
                json.dump(dist, f)
    with open(os.path.join(cfg.test.out_dir, "dim_results.json"), "w") as f:
        json.dump(all_results, f)

if __name__=="__main__":
    main()
