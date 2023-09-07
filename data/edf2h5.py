# example
# python3 edf2h5.py --in_file /storage/abarbu/full-day-ecog/m00191/m00191-full-part-4.EDF --out_dir m00191_part4--output_timestamp_metadata
from datetime import timedelta
import json
import time
import math
#import glob
import os
import h5py
import numpy as np
#from threading import Thread
import mne
from data_loading.utils import compute_m5_hash
from args import parse_edf2h5_args

def write_timestamp_metadata_file(file_name: str, out_dir=None, omit_prefix_percent=0) -> None:
    data = mne.io.read_raw_edf(file_name)
    info = data.info
    channels = data.ch_names
    timestamp = info["meas_date"]

    new_path = os.path.join(out_dir, 'time-stamp.json')
    idx = 0
    data_arr,_ = data[idx] #[1, n_samples]
    start_idx = int(omit_prefix_percent*data_arr.shape[-1])
    sample_rate = 2048
    timestamp + timedelta(seconds=start_idx/sample_rate)

    with open(new_path, 'w') as f:
        json.dump({'timestamp': f'{timestamp}'}, f)

def write_transposed_data_file(file_name: str, orig_channel_n, out_dir=None, omit_prefix_percent=0) -> None:
    data = mne.io.read_raw_edf(file_name)
    info = data.info
    channels = data.ch_names
    #raw_data = np.zeros([264,100])
    #channels = [f'C{i}' for i in range(164)] + [f'DC{i}' for i in range(264-163)]

    new_path = os.path.join(out_dir, 'data-sharded.h5')
    computed_hash = compute_m5_hash(file_name)

    with h5py.File(new_path, 'a', libver='latest') as hf_new:
        new_group = hf_new.create_group('data')
        hf_new['data'].attrs['orig_data_hash'] = computed_hash
        channel_count = 0
        orig_channel_n = orig_channel_n.tolist()
        for i in orig_channel_n:
            idx = i-1
            ch_name = channels[idx]
            print(idx, ch_name)
            data_arr,_ = data[idx] #[1, n_samples]
            start_idx = int(omit_prefix_percent*data_arr.shape[-1])
            data_arr = data_arr[0, start_idx:]
            data_arr = data_arr.squeeze()
            if ch_name in ['DC10', 'DC4']: #NOTE 
                new_group.create_dataset(ch_name, data=data_arr, compression="gzip")
                channel_count += 1
            else:
                new_group.create_dataset(f'electrode_{idx}', data=data_arr, compression="gzip")
                channel_count += 1

    return channel_count

if __name__=="__main__":
    start = time.time()
    args = parse_edf2h5_args()
    file = args.in_file
    subject = "m00191"
    trial = "trial000"

    headers_dir_format = '/storage/datasets/neuroscience/ecog/data-by-subject/{}/data/trials/{}/headers'

    def get_string_from_hdf5_reference(file, ref):
        return ''.join(chr(i) for i in file[ref[0]][:])

    headers_dir = headers_dir_format.format(subject, trial)
    header_file_name = os.listdir(headers_dir)[0]
    header_file = h5py.File(os.path.join(headers_dir, header_file_name),'r')
    electrode_labels = [get_string_from_hdf5_reference(header_file, ref) for ref in header_file['channel_labels']]
    labels = [(i, e) for i, e in enumerate(electrode_labels)]

    orig_channel_n = np.array(header_file['orig_channel_n']).squeeze()
    orig_channel_n = orig_channel_n.astype('int32')

    assert args.omit_prefix <= 1.0 and args.omit_prefix >= 0
    if args.output_timestamp_metadata:
        omit_prefix_percent = args.omit_prefix
        write_timestamp_metadata_file(file,  args.out_dir, omit_prefix_percent=omit_prefix_percent)
        exit()

    channel_count = write_transposed_data_file(file, orig_channel_n, args.out_dir, omit_prefix_percent=omit_prefix_percent)
    assert len(labels)==channel_count
    print(labels)
    print(file)
    total_time = time.time() - start
    print(f'that took {total_time/60}s')
