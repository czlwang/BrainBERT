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

def get_all_electrodes(subject):
    '''
        returns list of electrodes in this subject and trial
        NOTE: the order of these labels is important. Their position corresponds with a row in data.h5
    '''
    trial = "trial000" #Assume that every trial contains the same electrodes
    dataset_dir = "/storage/datasets/neuroscience/ecog"
    headers_dir = os.path.join(dataset_dir, f'data-by-subject/{subject}/data/trials/{trial}/headers')

    def get_string_from_hdf5_reference(f, ref):
        return ''.join(chr(i) for i in f[ref[0]][:])

    header_file_name = os.listdir(headers_dir)[0]
    header_file = h5py.File(os.path.join(headers_dir, header_file_name), 'r')
    electrode_labels = [get_string_from_hdf5_reference(header_file, ref) for ref in header_file['channel_labels']]

    strip_string = lambda x: x.replace("*","").replace("#","").replace("_","")
    electrode_labels = [strip_string(e) for e in electrode_labels]
    return electrode_labels

def clean_electrodes(subject, electrodes):
    corrupted_electrodes_path = "/storage/czw/self_supervised_seeg/data/corrupted_elec.json"
    with open(corrupted_electrodes_path, "r") as f:
        corrupted_elecs = json.load(f)
    corrupt = corrupted_elecs[subject]
    return list(set(electrodes).difference(corrupt))

def get_clean_laplacian_electrodes(subject):
    electrodes = get_all_electrodes(subject)
    electrodes = clean_electrodes(subject, electrodes)
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
