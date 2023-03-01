import hashlib
import os
import logging
from tqdm import tqdm as tqdm
import numpy as np
import csv
from pathlib import Path

log = logging.getLogger(__name__)

def file_as_bytes(file):
    with file:
        return file.read()

def compute_m5_hash(file_path):
    #from https://stackoverflow.com/a/3431835
    return hashlib.md5(file_as_bytes(open(file_path, 'rb'))).hexdigest()

def stem_electrode_name(name):
    #names look like 'O1aIb4', 'O1aIb5', 'O1aIb6', 'O1aIb7'
    #names look like 'T1b2
    reverse_name = reversed(name)
    found_stem_end = False
    stem, num = [], []
    for c in reversed(name):
        if c.isalpha():
            found_stem_end = True
        if found_stem_end:
            stem.append(c)
        else:
            num.append(c)
    return ''.join(reversed(stem)), int(''.join(reversed(num)))

def write_manifest(root_out, paths, lengths):
    absolute_path = Path(root_out).resolve()
    manifest_path = os.path.join(absolute_path, "manifests")
    Path(manifest_path).mkdir(exist_ok=True, parents=True)
    manifest_path = os.path.join(manifest_path, "manifest.tsv")
    with open(manifest_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow((str(absolute_path),))
        for row in zip(paths, lengths):
            writer.writerow(row)


