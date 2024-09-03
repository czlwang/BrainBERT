#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import pandas as pd
import numpy as np

# Define the function to create labels
def create_labels(file_path, events_dir):
    """
    Creates labels for each 5-second window of data based on seizure onset and offset times.

    Args:
        file_path (str): Path to the .npy file.
        events_dir (str): Path to the directory containing the events .tsv files.

    Returns:
        np.array: Array of labels.
    """
    # List to store the labels
    labels = []

    # Extract the base filename from the file path
    base_filename = os.path.basename(file_path).split('_ieeg')[0]

    # Determine if the file is ictal or interictal based on the filename
    if 'ictal' in base_filename and 'inter' not in base_filename:
        # The file is ictal
        label = 1

        # Construct the corresponding .tsv file path
        events_filename = base_filename + "_events.tsv"
        events_filepath = os.path.join(events_dir, events_filename)

        # Read the events file to get seizure onset and offset times
        events_df = pd.read_csv(events_filepath, sep='\t')

        # Extract the seizure onset and offset times
        seizure_onsets = events_df[events_df['trial_type'] == 'sz onset']['onset'].values
        seizure_offsets = events_df[events_df['trial_type'] == 'sz offset']['onset'].values

    elif 'interictal' in base_filename:
        # The file is interictal
        label = 0
    else:
        raise ValueError(f"Filename {base_filename} does not contain 'ictal' or 'interictal'")

    # Load the .npy file containing the data
    try:
        data = np.load(file_path)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return np.array([])  # Return an empty array on error

    # Number of rows in the data (each row is a 5-second window)
    num_rows = data.shape[0]
    window_size = 5  # Window size is 5 seconds

    if label == 1:
        # Create labels based on seizure onset and offset times for ictal files
        for i in range(num_rows):
            start_time = i * window_size  # Start time of the window
            end_time = start_time + window_size  # End time of the window
            label = 0  # Default to interictal

            # Check if the current window is within any seizure period
            for onset, offset in zip(seizure_onsets, seizure_offsets):
                if start_time <= offset and end_time >= onset:
                    label = 1  # Ictal period
                    break

            # Append the label for the current window
            labels.append(label)
    else:
        # For interictal files, all labels are 0
        labels.extend([label] * num_rows)

    # Convert labels to a numpy array
    labels = np.array(labels)

    # Return the labels as a numpy array
    return labels

# Example usage
#events_dir = '/home/vineetreddy/edf events'  # Directory containing the events .tsv files
#file_path = '/home/vineetreddy/edf numpy out/sub-HUP060_ses-presurgery_task-ictal_acq-seeg_run-01_ieeg_LAF1_concatenated.npy'  # Path to the concatenated .npy file
#labels = create_labels(file_path, events_dir)
#print(labels)


# In[ ]:




