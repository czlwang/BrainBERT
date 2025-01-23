from glob import glob as glob
from .utils import stem_electrode_name
import os
import json
import h5py

def get_all_laplacian_electrodes(elec_list):
    stems = [stem_electrode_name(e) for e in elec_list]
    def has_nbrs(stem, stems):
        (x,y) = stem
        return ((x,y+1) in stems) and ((x,y-1) in stems)
    laplacian_stems = [x for x in stems if has_nbrs(x, stems)]
    electrodes = [f'{x}{y}' for (x,y) in laplacian_stems]
    return electrodes

def get_all_electrodes(subject, data_root=None):
    '''
        returns list of electrodes in this subject and trial
        NOTE: the order of these labels is important. Their position corresponds with a row in data.h5
    '''
    electrode_labels_file = glob(os.path.join(data_root, "electrode_labels", subject, "electrode_labels.json"))
    assert len(electrode_labels_file)==1
    electrode_labels_file = electrode_labels_file[0]
    with open(electrode_labels_file, "r") as f:
        electrode_labels = json.load(f)
    strip_string = lambda x: x.replace("*","").replace("#","").replace("_","")
    electrode_labels = [strip_string(e) for e in electrode_labels]
    return electrode_labels

def clean_electrodes(subject, electrodes, data_root=None):
    corrupted_electrodes_path = os.path.join(data_root, "corrupted_elec.json")
    with open(corrupted_electrodes_path, "r") as f:
        corrupted_elecs = json.load(f)
    corrupt = corrupted_elecs[subject]
    return list(set(electrodes).difference(corrupt))

def get_clean_laplacian_electrodes(subject, data_root=None):
    electrodes = get_all_electrodes(subject, data_root=data_root)
    electrodes = clean_electrodes(subject, electrodes, data_root=data_root)
    laplacian_electrodes = get_all_laplacian_electrodes(electrodes)
    return laplacian_electrodes

def main():
    with open("data/pretrain_split_trials.json", "r") as f:
        subjects = json.load(f)
    all_electrodes = []
    for subject in subjects:
        electrodes = get_clean_laplacian_electrodes(subject)
        print(subject, len(electrodes))
        all_electrodes += [(subject, e) for e in electrodes]
    print(len(all_electrodes))

if __name__=="__main__":
    main()
