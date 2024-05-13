'''
this script takes in an audio file (or a folder of audio files) and outputs a trained neural network model that maps timbral features to spectra. 
steps:
1. load audio files
2. window the audio files
3. extract timbral features (MFCCs) per window
4. extract spectral features (STFT) per window
5. train a neural network to map timbral features to spectral features
6. save the model
'''

import os
from typing import List
import numpy as np
import librosa
import torch 
import torch.nn as nn
import torchaudio.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import math 

SAMPLE_RATE = 22050
WINDOW_SIZE = 1024
HOP_SIZE = WINDOW_SIZE // 4
use_pyin = False
use_torch_pitch = not use_pyin

def next_power_of_2(n):
    return 2**math.ceil(math.log2(n))

def num_evenly_fitting_hops(given_size, window_size, hop_size):
    # ceiling of the given size, to fit into an integral number of windows
    return 1 + int(np.ceil((given_size - window_size + 1) / hop_size))

# Step 1: Load audio files. Should work with either single file or whole folder.
def load_audio_files(audio_path):
    '''
    Load audio files from a given path. 
    Args:
        audio_path (str): path to audio file or folder of audio files
    Returns:
        audio_data (List): List of audio data
    '''
    # use torchaudio to load audio files
    
    if os.path.isdir(audio_path):
        audio_files = [os.path.join(audio_path, f) for f in os.listdir(audio_path) if f.endswith('.wav')]
        audio_data = [librosa.load(f, sr=SAMPLE_RATE, mono=True)[0] for f in audio_files]
    else:
        audio_data = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)[0]
        audio_data = [audio_data]
    return audio_data

# Step 2: Window the audio files, zero-pad if necessary

# test
audio_data = load_audio_files('../audio')

mfccs = [librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=13, 
                              n_fft=WINDOW_SIZE, hop_length=HOP_SIZE) 
                              for y in audio_data]
mfccs = np.concatenate(mfccs, axis=1)

stft = [np.abs(librosa.stft(y, n_fft=WINDOW_SIZE, hop_length=HOP_SIZE))
         for y in audio_data]
stft = np.concatenate(stft, axis=1)

inversed = librosa.istft(stft, win_length = WINDOW_SIZE , hop_length=HOP_SIZE)
# get fundamental frequencies for each window
if use_pyin:
    f0, voiced_flag, voiced_prob = librosa.pyin(inversed, 
             fmin=200, fmax=1200, 
             sr=SAMPLE_RATE, 
             frame_length=WINDOW_SIZE, 
             win_length=WINDOW_SIZE//2, 
             hop_length=HOP_SIZE, 
             n_thresholds=100, 
             beta_parameters=(2, 18), 
             boltzmann_parameter=2, 
             resolution=0.1, # in semitones
             max_transition_rate=35.92, 
             switch_prob=0.01, 
             no_trough_prob=0.01, 
             fill_na=None,  # fill unvoiced values with this. if None, will be best guess.
             center=False, 
             pad_mode='constant')   # ignored when center=False
elif use_torch_pitch:
    # use torchaudio to compute pitch
    audio_tensor = torch.tensor(inversed).unsqueeze(0)
    pitch = F.detect_pitch_frequency(audio_tensor, sample_rate=SAMPLE_RATE, 
                         frame_time=WINDOW_SIZE/SAMPLE_RATE, win_length=WINDOW_SIZE,
                         freq_low=200, freq_high=1200)
    f0 = pitch[0].numpy()
else:
    raise ValueError("Must use either pyin or torchaudio pitch detection")


