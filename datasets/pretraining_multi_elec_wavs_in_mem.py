#python3 -m data.create_data_dirs +data=pretraining_subject3 +hydra.job.chdir=False
from data.electrode_subject_data import ElectrodeSubjectData
from torch.utils import data
import numpy as np
from datasets import register_dataset

def get_electrode_subj_data(args):
    s = ElectrodeSubjectData(args.subject, args)
    return s

@register_dataset(name="pretraining_wavs_multi_elec_in_mem")
class PretrainingMultiElecWavsInMem(data.Dataset):
    #NOTE: this is to be used in pre-training, while the other class in this file is to be used during fine-tuning
    #Takes multiple electrodes and makes each one its own example
    def __init__(self, args) -> None:
        subject_data = get_electrode_subj_data(args)
        self.subject_data = subject_data
        self.seeg_data = subject_data.neural_data
        self.seeg_data = np.transpose(self.seeg_data, [1,0,2])
        self.seeg_data = self.seeg_data.reshape([-1, self.seeg_data.shape[-1]])

    def __len__(self):
        '''
            returns:
                Number of words in the dataset
        '''
        return self.seeg_data.shape[0]

    def __getitem__(self, idx: int):

        #NOTE: remember not to load to cuda here
        target = self.seeg_data[idx]
        return {
                "input" : target,
               }
