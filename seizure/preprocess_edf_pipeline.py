#!/usr/bin/env python
# coding: utf-8

# # SEEG Data Processing Script
# 
# This script processes Stereo-Electroencephalography (SEEG) data stored in EDF files. It includes several functions to create output directories, filter SEEG channels, and process the data into epochs. The processed data is then saved as numpy arrays for further analysis.
# 
# ## Code Overview
# 
# ### Importing Libraries
# 
# The script begins by importing necessary libraries, including `os` for file handling, `numpy` for numerical operations, and `mne` for processing EEG data.
# 
# ### Function Definitions
# 
# #### 1. Creating Output Directory
# 
# A function is defined to create an output directory if it doesn't already exist. This ensures that the subsequent steps have a designated place to save processed data.
# 
# #### 2. Getting SEEG Channels
# 
# This function identifies SEEG channels from the raw data. It checks for SEEG and EEG labels, excluding standard 10-20 scalp EEG channel names and EKG channels, to isolate the SEEG channels.
# 
# #### 3. Processing a Single File
# 
# This function processes a single EDF file by performing the following steps:
# - Loads the raw SEEG data.
# - Filters the data to include only SEEG channels.
# - Applies a high-pass filter at 0.1 Hz and resamples the data to 256 Hz.
# - Applies a notch filter to remove harmonics of 60 Hz below the Nyquist frequency.
# - Segments the data into 5-second epochs.
# - Saves the filtered data as `.npy` files for each channel and epoch.
# 
# #### 4. Concatenating Epochs
# 
# A function is defined to concatenate the saved epochs for each channel. It sorts the epochs by their index, concatenates them, and saves the concatenated data for each channel as `.npy` files.
# 
# ### Main Function to Process the Entire Directory
# 
# The main function processes all EDF files in a specified directory. It:
# - Creates the necessary output directories.
# - Processes each EDF file to create epochs.
# - Concatenates the epochs for each channel.
# 
# After processing, the concatenated data is saved for further analysis.
# 

# In[1]:


# Import necessary libraries
#!pip install mne
import os
import numpy as np
import mne

# Define functions

# Function to create output directory
def create_output_directory(output_dir):
    """
    Creates an output directory if it does not already exist.

    Parameters:
    output_dir (str): The path to the directory to be created.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

# Function to get SEEG channels
def get_seeg_channels(raw):
    """
    Identifies and returns SEEG channels from the given MNE raw object.

    Parameters:
    raw (mne.io.Raw): The MNE raw object containing the EEG data.

    Returns:
    list: A list of SEEG channel names.
    """
    # Define standard 10-20 scalp EEG channel names (in lowercase)
    scalp_eeg_1020 = [
        "fp1", "fp2", "f7", "f3", "fz", "f4", "f8", "t7", "c3", "cz", "c4", "t8",
        "p7", "p3", "pz", "p4", "p8", "o1", "o2", "t3", "t4", "t5", "t6",
    ]

    # Check for SEEG and EEG labels
    seeg_chs = mne.pick_types(raw.info, meg=False, eeg=False, seeg=True)
    eeg_chs = mne.pick_types(raw.info, meg=False, eeg=True, seeg=False)

    # List to store SEEG channels
    seeg_channels = []

    if len(seeg_chs) > 0:
        seeg_channels = [raw.ch_names[idx] for idx in seeg_chs]
    elif len(eeg_chs) > 0:
        for idx in eeg_chs:
            ch_name = raw.ch_names[idx]
            if ch_name.lower() not in scalp_eeg_1020 and "ekg" not in ch_name.lower():
                seeg_channels.append(ch_name)
    else:
        # If no SEEG or EEG labels, check channel labels
        for ch_name in raw.ch_names:
            if ch_name.lower() not in scalp_eeg_1020 and "ekg" not in ch_name.lower():
                seeg_channels.append(ch_name)

    return seeg_channels

# Function to process a single file
def process_file(filepath, epoch_output_dir):
    """
    Processes a single EDF file to filter, resample, and create 5-second epochs of SEEG data.

    Parameters:
    filepath (str): The path to the EDF file to be processed.
    epoch_output_dir (str): The directory where the epoch .npy files will be saved.
    """
    original_file_name = os.path.splitext(os.path.basename(filepath))[0]
    
    try:
        # Load the raw data
        raw = mne.io.read_raw_edf(filepath, preload=True)

        # Get SEEG channels
        seeg_channels = get_seeg_channels(raw)

        # Create a copy of raw and pick only SEEG channels
        raw_filter = raw.copy().pick_channels(seeg_channels)

        # Apply the high-pass filter at 0.1 Hz
        raw_filter.filter(l_freq=0.1, h_freq=None, picks="all")
        
        # Resample the data to 256 Hz
        raw_filter.resample(256, npad="auto")

        # Get the sampling rate (sfreq) and calculate the Nyquist frequency
        sfreq = raw_filter.info["sfreq"]
        nyquist_freq = sfreq / 2.0

        # Generate the list of frequencies to remove (i.e., harmonics of 60 Hz below the Nyquist frequency)
        base_freq = 60
        freqs = []
        current_freq = base_freq
        while current_freq < nyquist_freq:
            freqs.append(current_freq)
            current_freq += base_freq

        # Apply the notch filter
        raw_filter.notch_filter(freqs=freqs, picks="all")

        # Create 5-second epochs
        epochs = mne.make_fixed_length_epochs(raw_filter, duration=5.0, preload=True)

        # Save filtered data to .npy files for each channel and epoch. Note that epoch index starts at 1.
        for channel_idx, channel_name in enumerate(seeg_channels):
            # Get data for the specific channel across all epochs
            data = epochs.get_data(picks=[channel_name])
            for epoch_idx in range(data.shape[0]):
                epoch_data = data[epoch_idx, 0, :]
                output_path = os.path.join(
                    epoch_output_dir, f"{original_file_name}_{channel_name}_epoch{epoch_idx+1}.npy"
                )
                np.save(output_path, epoch_data)

    except Exception as e:
        print(f"Error processing file {filepath}: {e}")

def concatenate_epochs(epoch_output_dir, concat_output_dir):
    """
    Concatenates epochs for each channel from .npy files and saves the concatenated data.

    Parameters:
    epoch_output_dir (str): The directory containing the epoch .npy files.
    concat_output_dir (str): The directory where the concatenated .npy files will be saved.
    """
    create_output_directory(concat_output_dir)
    
    # Initialize a dictionary to hold the concatenated epochs for each channel
    channels_data = {}

    # Iterate over the files in the directory
    for filename in os.listdir(epoch_output_dir):
        if filename.endswith('.npy'):
            filepath = os.path.join(epoch_output_dir, filename)
            # Extract the channel name by removing the epoch part of the filename
            channel_name = filename.split('_epoch')[0]
            # Extract the epoch number from the file name
            epoch_number = int(filename.split('_epoch')[1].split('.')[0])
            
            # Load the numpy array from the file
            epoch_data = np.load(filepath)
            
            # If the channel is not already in the dictionary, add it
            if channel_name not in channels_data:
                channels_data[channel_name] = []
            
            # Append a tuple of the epoch number and epoch data to the list for the corresponding channel
            channels_data[channel_name].append((epoch_number, epoch_data))

    # Sort the epochs for each channel by epoch number and concatenate them
    for channel, epochs in channels_data.items():
        # Sort the epochs based on the epoch number
        epochs.sort(key=lambda x: x[0])
        # Extract only the epoch data (sorted)
        sorted_epoch_data = [epoch[1] for epoch in epochs]
        # Concatenate the sorted epoch data
        concatenated_data = np.vstack(sorted_epoch_data)
        # Save the concatenated data for each channel
        np.save(os.path.join(concat_output_dir, f'{channel}_concatenated.npy'), concatenated_data)
        # Print the shape of the concatenated data
        #print(f'Channel: {channel}, Concatenated Data Shape: {concatenated_data.shape}')
    
    print("Data processing complete. Concatenated files saved.")

# Main function to process the entire directory
def process_directory(input_dir, epoch_output_dir, concat_output_dir):
    """
    Processes all EDF files in a directory by creating epochs and concatenating them for each channel.

    Parameters:
    input_dir (str): The directory containing the EDF files to be processed.
    epoch_output_dir (str): The directory where the epoch .npy files will be saved.
    concat_output_dir (str): The directory where the concatenated .npy files will be saved.
    """
    create_output_directory(epoch_output_dir)
    create_output_directory(concat_output_dir)
    
    # Step 1: Process EDF files to create epochs
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".edf"):
                filepath = os.path.join(root, file)
                process_file(filepath, epoch_output_dir)
    
    # Step 2: Concatenate epochs for each channel
    concatenate_epochs(epoch_output_dir, concat_output_dir)


# ## Example Usage

# In[ ]:


# Example Usage
# Define paths and process the directory
#input_dir = "/home/vineetreddy/edf"
#epoch_output_dir = "/home/vineetreddy/edf numpy"
#concat_output_dir = "/home/vineetreddy/edf numpy out"
#process_directory(input_dir, epoch_output_dir, concat_output_dir)

