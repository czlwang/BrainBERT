#to write features to disk
#python3 -m data.write_preprocessed_inputs +data=tf_unmasked +data_prep=cwt_to_disk ++data.data=all_day_data
from multiprocessing import Process
from omegaconf import DictConfig, OmegaConf
import hydra
import models
import tasks
from runner import Runner
from datasets import build_dataset
import logging
import os
from pathlib import Path
import numpy as np
from tqdm import tqdm as tqdm
import csv
import yaml
import preprocessors

#take dataset with wavs and write it to cache
log = logging.getLogger(__name__)

def singleprocess_save_pretrained_spec_cache(pid, idxs, cache_path, seeg_data, extracter, extracter_cache_path, read_spec_from_cache=True):
    #save features for the case where you have the spectrograms cached, and now need to push them through the pretrained model
    for idx in tqdm(idxs):
        if read_spec_from_cache:
            spec = np.load(os.path.join(extracter_cache_path, f"{idx}.npy"))
        else:
            spec = extracter.spec_preprocessor(seeg_data[idx])
        item = extracter(seeg_data[idx], spec_preprocessed=spec)
        save_path = os.path.join(cache_path, f'{idx}.npy')
        np.save(save_path, item.numpy())

def singleprocess_save_cache(pid, idxs, cache_path, seeg_data, extracter):
    for idx in tqdm(idxs):
        item = extracter(seeg_data[idx])
        save_path = os.path.join(cache_path, f'{idx}.npy')
        np.save(save_path, item.numpy())

def save_cache(idxs, cache_path, seeg_data, extracter):
    if isinstance(extracter, preprocessors.spec_pretrained.SpecPretrained):
        extracter_cache_path = os.path.join(cache_path, "cached_spec")
        Path(extracter_cache_path).mkdir(exist_ok=True, parents=True)
        if not isinstance(extracter.spec_preprocessor, preprocessors.stft.STFTPreprocessor):
            multiprocess_save_cache(idxs, extracter_cache_path, seeg_data, extracter.spec_preprocessor)
            singleprocess_save_pretrained_spec_cache(0, idxs, cache_path, seeg_data, extracter, extracter_cache_path)
        else:
            singleprocess_save_pretrained_spec_cache(0, idxs, cache_path, seeg_data, extracter, extracter_cache_path, read_spec_from_cache=False)
    else:
        multiprocess_save_cache(idxs, cache_path, seeg_data, extracter)

def multiprocess_save_cache(idxs, cache_path, seeg_data, extracter):
    log.info("Writing data to disk")

    ps = []
    n=32
    step = int(len(idxs)/n) + 1
    ranges = [(i*step, (i+1)*step) for i in range(n)]
    for index in range(n):
        start, end = ranges[index]
        slice_idxs = idxs[start:end]
        log.info(f'Main    : create and start process {index} with {start} to {end}')
        pid = index
        x = Process(target=singleprocess_save_cache, args=(pid, slice_idxs, cache_path, seeg_data, extracter))
        ps.append(x)
        x.start()

    for index, process in enumerate(ps):
        log.info(f'Main    : before joining process {index}')
        process.join()
        log.info("Main    : process %d done", index)
