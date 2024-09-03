#!/usr/bin/env python
# coding: utf-8

# # BrainBERT Seizure Detection with Logistic Regression Pipeline
# 
# This script is designed to import necessary libraries and modules, set up the working environment, load a pre-trained BrainBERT model, process EEG data to generate embeddings, and train a logistic regression model for seizure detection. 
# 
# 1. **Import Libraries and Modules**: The script begins by importing essential libraries and modules, such as `numpy` for numerical operations, `matplotlib` for plotting, `scipy` for signal processing, `torch` for deep learning, `omegaconf` for configuration management, and `sklearn` for machine learning tasks.
# 
# 2. **Set Working Directory**: The working directory is set to the BrainBERT directory, and the parent directory is added to the system path to ensure all custom modules can be imported.
# 
# 3. **Import Custom Functions and Models**: Custom functions and models specific to the BrainBERT project are imported. These include functions for creating labels and handling the BrainBERT model.
# 
# 4. **Load Pre-Trained Model**: A function `load_brainbert_model` is defined to load the BrainBERT model with pre-trained weights. It uses a configuration file and checkpoint path to build and initialize the model on a GPU.
# 
# 5. **Generate BrainBERT Embeddings**: A function `generate_brainbert_embeddings` is defined to process example waveforms and generate BrainBERT embeddings. This involves calculating the Short-Time Fourier Transform (STFT) of the signal, preparing inputs for the model, and obtaining model outputs.
# 
# 6. **Load Pre-Trained Weights**: The script loads pre-trained weights for the BrainBERT model from a specified checkpoint path.
# 
# 7. **Process EEG Data**: The script iterates over `.npy` files in a specified directory, loading the example waveforms and generating corresponding labels. It then generates BrainBERT embeddings for each example and stores the results.
# 
# 8. **Combine Data**: All generated BrainBERT embeddings and labels are concatenated into single arrays.
# 
# 9. **Train-Test Split**: The data is split into training and testing sets using an 80-20 split to ensure reproducibility.
# 
# 10. **Train Logistic Regression Model**: A logistic regression model is trained using the training data. The trained model is then saved to a specified path.
# 
# 11. **Evaluate Model**: The logistic regression model predicts labels for the test set, and the accuracy of the predictions is calculated and printed.
# 
# This script provides a comprehensive workflow for processing EEG data, generating embeddings using a pre-trained BrainBERT model, and training a machine learning model for seizure detection.
# 

# In[ ]:


# Import Libraries and Modules
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, stats
import torch
from omegaconf import OmegaConf
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib

# Set the working directory to the BrainBERT directory
os.chdir('/home/vineetreddy/Dropbox/CZW_MIT/BrainBERT')
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)  # Add the parent directory to the system path

# Import custom functions and models
from demo_brainbert_annotated import *  # Importing custom functions
from create_labels import create_labels  # Import custom function to create labels for seizure data
import models  # Import custom models (user-defined)

# Function to load pre-trained model weights and configuration
def load_brainbert_model(ckpt_path):
    """
    Loads the BrainBERT model with pre-trained weights.

    Args:
        ckpt_path (str): Path to the checkpoint file.

    Returns:
        torch.nn.Module: BrainBERT model with loaded weights.
    """
    cfg = OmegaConf.create({"upstream_ckpt": ckpt_path})
    brainbert_model = build_model(cfg)  # Build the model with the given configuration
    brainbert_model.to('cuda')  # Move the model to GPU
    init_state = torch.load(ckpt_path)  # Load the initial state of the model
    load_model_weights(brainbert_model, init_state['model'], False)  # Load the model weights
    return brainbert_model

# Function to generate BrainBERT embeddings from example waveforms
def generate_brainbert_embeddings(model, example_wavs):
    """
    Generates BrainBERT embeddings for each example.

    Args:
        brainbert_model (torch.nn.Module): BrainBERT model.
        example_wavs (np.array): Array of example waveforms.

    Returns:
        np.array: Array of BrainBERT embeddings.
    """
    brainbert_outs = []
    for example_wav in example_wavs:
        # Get the Short-Time Fourier Transform (STFT) of the signal
        f, t, linear = get_stft(example_wav, 2048, clip_fs=25, nperseg=400, noverlap=350, normalizing="zscore", return_onesided=True)
        inputs = torch.FloatTensor(linear).unsqueeze(0).transpose(1, 2).to('cuda')  # Prepare inputs for the model
        mask = torch.zeros((inputs.shape[:2])).bool().to('cuda')  # Create a mask for the inputs
        with torch.no_grad():
            out = brainbert_model.forward(inputs, mask, intermediate_rep=True)  # Get the model output
        brainbert_outs.append(out.cpu().numpy())  # Append the output to the list

    # Concatenate and average the outputs
    brainbert_outs_arrr = np.concatenate(brainbert_outs, axis=0)
    brainbert_outs_arr = brainbert_outs_arrr.mean(axis=1)
    
    return brainbert_outs_arr

# Load Pre-Trained Model Weights and Configuration
ckpt_path = "/home/vineetreddy/Dropbox/CZW_MIT/stft_large_pretrained_256hz.pth"  # Path to pre-trained weights for the model
brainbert_model = load_brainbert_model(ckpt_path)  # Load the model

# Directory paths
directory = '/home/vineetreddy/edf numpy out/'  # Directory containing .npy files
events_dir = '/home/vineetreddy/edf events'  # Directory containing the events .tsv files
all_brainbert_outs = []
all_labels = []

# Process each file in the directory
for filename in os.listdir(directory):
    if filename.endswith(".npy"):
        file_path = os.path.join(directory, filename)
        
        # Load in channel array. Each channel array is organized such that each row is a 5-second window 
        # and the columns are the time series data
        example_wavs = np.load(file_path)

        # Generate labels
        labels = create_labels(file_path, events_dir)
        if labels.size == 0:
            continue

        # Generate BrainBERT embeddings for each example
        brainbert_outs_arr = generate_brainbert_embeddings(brainbert_model, example_wavs)
        all_brainbert_outs.append(brainbert_outs_arr)
        all_labels.append(labels)

# Combine all the data
all_brainbert_outs = np.concatenate(all_brainbert_outs, axis=0)
all_labels = np.concatenate(all_labels, axis=0)

# Split the data into training and testing sets
# 80% of the data for training and 20% for testing; random_state=42 ensures reproducibility of the split
brainbert_outs_arr_train, brainbert_outs_arr_test, labels_train, labels_test = train_test_split(
    all_brainbert_outs, all_labels, test_size=0.2, random_state=42)

# Logistic Regression
logistic_model = LogisticRegression()
logistic_model.fit(brainbert_outs_arr_train, labels_train)  # Fit the logistic regression model

# Save the logistic regression model
model_save_path = "/home/vineetreddy/save model/logistic_model.joblib"
joblib.dump(logistic_model, model_save_path)
print(f"Model saved to {model_save_path}")

# Predict the labels
predictions = logistic_model.predict(brainbert_outs_arr_test)  # Predict the labels
acc = np.mean(predictions == labels_test)
print(f"Accuracy: {acc}")


# In[ ]:




