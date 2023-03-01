import torch
import random
import numpy as np

def get_mask_fill_value(data):
    return 0
    #return data.min() - 1

def create_masked_intervals(data, consecutive_min, consecutive_max, mask_p, axis="time"):
    consecutive_min = consecutive_min
    consecutive_max = consecutive_max
    assert consecutive_min <= consecutive_max
    assert consecutive_max < data.shape[0]

    if axis=="time":
        valid_starts = range(len(data)-consecutive_max)
    else:
        valid_starts = range(data.shape[1]-consecutive_max) 
    masked_steps = []
    for i in valid_starts:
        if random.random() < mask_p:
            if len(masked_steps)==0 or masked_steps[-1][1] < i:
                masked_steps.append((i, i+random.randint(consecutive_min, consecutive_max)))
    #masked_steps = [i for i in valid_starts if random.random() < mask_p]
    #masked_steps = [(i, i+random.randint(consecutive_min, consecutive_max)) for i in masked_steps]
    return masked_steps, valid_starts

def fixed_mask_inputs(data, task_cfg):
    mask_label = torch.zeros_like(data)

    masked_steps, valid_starts = create_masked_intervals(data, task_cfg.time_mask_consecutive_min, task_cfg.time_mask_consecutive_max, task_cfg.time_mask_p, axis="time")
    for (start,end) in masked_steps:
        mask_label[start:end,:] = 1

    masked_data = torch.clone(data)
    mask_fill_value = get_mask_fill_value(data)
    for (start,end) in masked_steps:
        dice = random.random()
        if dice < 0.1:#TODO look at attentions
            pass
        elif dice < 0.2:
            random_replace_start = random.randint(0, len(valid_starts)-1)
            diff = end-start
            masked_data[start:end,:] = data[random_replace_start:random_replace_start+diff,:]
        else:
            masked_data[start:end,:] = mask_fill_value

    for (start,end) in masked_steps:
        mask_label[:,start:end] = 1

    masked_steps, valid_starts = create_masked_intervals(data, task_cfg.freq_mask_consecutive_min, task_cfg.freq_mask_consecutive_max, task_cfg.freq_mask_p, axis="freq")
    for (start,end) in masked_steps:
        dice = random.random()
        if dice < 0.1:#TODO look at attentions
            pass
        elif dice < 0.2:
            random_replace_start = valid_starts[random.randint(0, len(valid_starts)-1)]
            diff = end-start
            masked_data[:,start:end] = data[:,random_replace_start:random_replace_start+diff]
        else:
            masked_data[:,start:end] = mask_fill_value
    return masked_data, mask_label

def variable_mask_time(data, task_cfg):
    decim = 60
    sample_rate = 2048
    max_size_in_secs = 0.250
    max_size_in_samples = max_size_in_secs*sample_rate/decim

    min_size_in_samples = random.randint(1,2)

    fs = np.linspace(task_cfg.min_f, task_cfg.max_f, task_cfg.n_freq_steps)
    window_sizes = [int(max(min_size_in_samples,200/(25+f))) for f in fs]
    #window_sizes = [int(max(0,160/(30+f))) for f in fs]

    max_size = max(window_sizes)
    valid_starts = list(np.arange(max_size, data.shape[0] - max_size)) #remember that mask is centered on time position

    def fill_in_time_mask(array, position, value=None, value_slice=None):
        #value -- what value to fill the template with
        assert not (value != None and value_slice != None)
        arr_len = array.shape[0]
        if value_slice is not None:
            for i in range(len(window_sizes)):
                array[max(0,position-window_sizes[i]):min(arr_len,position+window_sizes[i]),i] = value_slice[i]
        else:
            for i in range(len(window_sizes)):
                array[max(0,position-window_sizes[i]):min(arr_len,position+window_sizes[i]),i] = value
        return array

    def take_time_mask(array, position):
        arr_len = array.shape[0]
        value_slice = []
        for i in range(len(window_sizes)):
            value_slice.append(array[max(0,position-window_sizes[i]):min(arr_len,position+window_sizes[i]), i])
        return value_slice

    masked_positions = []
    max_window = 2*max(window_sizes)
    for pos in valid_starts:
        if random.random() < task_cfg.mask_p:
            if len(masked_positions)==0 or abs(masked_positions[-1] - pos) > max_window+1:
                masked_positions.append(pos)
    #import pdb; pdb.set_trace()
    masked_data = torch.clone(data)
    mask_label = torch.zeros_like(data)
    for position in masked_positions:
        dice = random.random()
        if dice < 0.1:#TODO look at attentions
            pass
        elif dice < 0.2:
            random_position = valid_starts[random.randint(0, len(valid_starts)-1)]
            value_slice = take_time_mask(data, random_position)
            masked_data = fill_in_time_mask(masked_data, position, value_slice=value_slice)
        else:
            mask_fill_value = get_mask_fill_value(data)
            masked_data = fill_in_time_mask(masked_data, position, value=mask_fill_value)
        mask_label = fill_in_time_mask(mask_label, position, 1)
    return masked_data, mask_label

def variable_mask_freq(data, task_cfg):
    fs = np.linspace(task_cfg.min_f, task_cfg.max_f, task_cfg.n_freq_steps)
    #mask_sizes = list(reversed([max(1,int(max(0,160/(30+f)))) for f in fs]))
    mask_sizes = [max(1,int(4.9*(f)/250)) for f in fs]
    idx2mask_size = {i:s for i,s in enumerate(mask_sizes)}
    valid_starts = list(range(data.shape[1] - max(mask_sizes)))
    masked_positions = [i for i in valid_starts if random.random() < task_cfg.mask_p]

    masked_data = torch.clone(data)
    mask_label = torch.zeros_like(data)

    mask_fill_value = get_mask_fill_value(data)
    for position in masked_positions:
        diff = idx2mask_size[position]
        dice = random.random()
        if dice < 0.1:#TODO look at attentions
            pass
        elif dice < 0.2:
            random_replace_start = valid_starts[random.randint(0, len(valid_starts)-1)]
            masked_data[:,position:position+diff] = data[:,random_replace_start:random_replace_start+diff]
        else:
            masked_data[:,position:position+diff] = mask_fill_value
        mask_label[:,position:position+diff] = 1
    return masked_data, mask_label

def variable_mask(data, task_cfg):
    masked_data, mask_label = variable_mask_time(data, task_cfg)

    masked_data, freq_mask_label = variable_mask_freq(masked_data, task_cfg)
    mask_label += freq_mask_label
    return masked_data, mask_label

def mask_inputs(data, task_cfg):
    if task_cfg.mask_type=="fixed":
        return fixed_mask_inputs(data, task_cfg)
    elif task_cfg.mask_type=="variable":
        return variable_mask(data, task_cfg)
