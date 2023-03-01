from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

def split_dataset(dataset, args):
    val_split = args.get("val_split", 0)
    test_split = args.get("test_split", 0)
    train_split = args.get("train_split", 1-val_split-test_split)
    assert val_split + test_split + train_split <= 1
    assert train_split > 0
    all_idxs = list(range(len(dataset))) 
    train_idxs, test_val_idxs = train_test_split(all_idxs, test_size=val_split+test_split, random_state=42)   
    train_idxs = train_idxs[:int(len(all_idxs)*train_split)]
    train_fewshot = args.get("train_fewshot", -1)
    train_idxs = train_idxs[:train_fewshot]

    train_set = Subset(dataset, train_idxs)

    val_idxs, test_idxs = train_test_split(test_val_idxs, test_size=test_split/(val_split+test_split), random_state=42)   
    val_set = Subset(dataset, val_idxs)
    test_set = Subset(dataset, test_idxs)
    return train_set, val_set, test_set

