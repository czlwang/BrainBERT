from sklearn.model_selection import train_test_split
import logging
from pathlib import Path
import os
import json

log = logging.getLogger(__name__)

def get_split_idxs(dataset, args):
    val_split = args.get("val_split", 0)
    test_split = args.get("test_split", 0)
    train_split = args.get("train_split", 1-val_split-test_split)
    assert val_split + test_split + train_split <= 1
    assert train_split > 0
    all_idxs = list(range(len(dataset))) 
    train_idxs, test_val_idxs = train_test_split(all_idxs, test_size=val_split+test_split, random_state=42)   
    #train_idxs = train_idxs[:int(len(all_idxs)*train_split)] #NOTE: I'm pretty sure this is a holdover from before I used train_test_split
    train_fewshot = args.get("train_fewshot", len(train_idxs))
    train_idxs = train_idxs[:train_fewshot]

    val_idxs, test_idxs = train_test_split(test_val_idxs, test_size=test_split/(val_split+test_split), random_state=42)   
    return train_idxs, val_idxs, test_idxs

def make_saved_data_split_path_name(args):
    '''
    saved_data_split is the root path to all saved data splits
    assumes that the only thing relevant to the data split is the subject, trials, and dataset name
    '''
    #NOTE: previously, we used to generate the save_dir automatically. Now, we make the user specify.
    #trials = '_'.join(args.brain_runs)
    #return os.path.join(args.saved_data_split, args.name, args.subject, trials)
    return args.saved_data_split

def can_load_saved_splits(args):
    if 'saved_data_split' in args: 
        saved_data_split_path = make_saved_data_split_path_name(args)
        if os.path.exists(saved_data_split_path):
            try:
                with open(os.path.join(saved_data_split_path,"splits.json"), "r") as f:
                    idxs = json.load(f)
                return True
            except:
                return False
    return False

def split_dataset_idxs(dataset, args):
    if can_load_saved_splits(args):
        saved_data_split_path = make_saved_data_split_path_name(args)
        log.info(f"Using saved train/val/test split at {saved_data_split_path}")
        with open(os.path.join(saved_data_split_path,"splits.json"), "r") as f:
            idxs = json.load(f)
        train_idxs, val_idxs, test_idxs = idxs["train"], idxs["val"], idxs["test"]
    else:
        log.info(f"Creating train/val/test split")
        train_idxs, val_idxs, test_idxs = get_split_idxs(dataset, args)

    if 'saved_data_split' in args and not can_load_saved_splits(args):
        saved_data_split_path = make_saved_data_split_path_name(args)
        log.info(f"Saving train/val/test split {saved_data_split_path}")
        Path(saved_data_split_path).mkdir(exist_ok=True, parents=True)
        with open(os.path.join(saved_data_split_path,"splits.json"), "w") as f:
            json.dump({"train": train_idxs, "val": val_idxs, "test": test_idxs}, f)

    #assert len(train_idxs) + len(val_idxs) + len(test_idxs) == len(dataset) #no longer needed, because can be smaller than the total

    return train_idxs, val_idxs, test_idxs