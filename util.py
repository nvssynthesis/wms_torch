import os
import torchaudio
import torch
import math 
import numpy as np
import torchaudio.transforms as T

def process_wave(wave: np.ndarray, current_fs: float, desired_fs: float):
    resampler = T.Resample(current_fs, desired_fs, dtype=wave.dtype)
    return resampler(wave)

def load_audio_files(audio_path, desired_sample_rate) -> np.array:
    '''
    Load audio files from a given path. 
    Args:
        audio_path (str): path to audio file or folder of audio files
    Returns:
        audio_data (List): List of audio data
    '''

    if os.path.isdir(audio_path):
        audio_files = [os.path.join(audio_path, f) for f in os.listdir(audio_path) if f.endswith('.wav')]
        audio_data = []
        for f in audio_files:
            wave, fs = torchaudio.load(f, normalize=True)
            audio_data.append(process_wave(wave, fs, desired_sample_rate))
    else:
        wave, fs = torchaudio.load(f, normalize=True)
        audio_data = [process_wave(wave, fs, desired_sample_rate)]
    return audio_data

def make_sequences(data: torch.Tensor, sequence_length: int) -> torch.Tensor:
    '''
    Make sequences of data with a given sequence length
    Args:
        data (Tensor): Data
        sequence_length (int): Sequence length
    Returns:
        sequences (Tensor): Sequences of data
    '''
    sequences = []
    for i in range(data.shape[0] - sequence_length + 1):
        sequences.append(data[i:i+sequence_length])

    sequences = torch.stack(sequences)
    return sequences


def listOfWaveformsToTensor(waveforms: list[np.ndarray], window_size) -> torch.Tensor:
    '''
    Concatenates a list of waveforms into a single tensor, zero-padding to make each a multiple of window_size
    Args:
        waveforms (List): List of waveforms
    Returns:
        audio_tensor (Tensor): Tensor with rows of waveforms each of window_size
    '''
    padded_waveforms = []
    for w in waveforms:
        pad_amt = window_size - w.shape[1] % window_size
        padded_waveform = np.pad(w, [(0,0), (0, pad_amt)]) 
        padded_waveforms.append(padded_waveform)
    # concatenate the waveforms into a single tensor
    audio_tensor = torch.tensor(np.concatenate(padded_waveforms, axis=1))
    # reshape the tensor so that each row is a window
    audio_tensor = audio_tensor.view(-1, window_size)
    return audio_tensor

def concatenateWaveforms(waveforms: list[np.ndarray], window_size) -> np.ndarray:
    padded_waveforms = []
    for w in waveforms:
        pad_amt = window_size - w.shape[1] % window_size
        padded_waveform = np.pad(w, [(0,0), (0, pad_amt)]) 
        padded_waveforms.append(padded_waveform)
    # concatenate the waveforms into a single tensor
    audio_tensor = torch.tensor(np.concatenate(padded_waveforms, axis=1))
    # reshape the tensor so that each row is a window
    return audio_tensor

def next_power_of_2(n):
    return 2**math.ceil(math.log2(n))

def num_evenly_fitting_hops(given_size, window_size, hop_size):
    # ceiling of the given size, to fit into an integral number of windows
    return 1 + int(np.ceil((given_size - window_size + 1) / hop_size))

def make_conjugate_symmetric(x: torch.Tensor) -> torch.Tensor:
    '''
    used to make the input to the ifft result in a real signal.
    '''
    conj = x[1:-1].conj()
    conj = conj.flip(dims=[0])
    result = torch.cat([x, conj])
    return result