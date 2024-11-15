import os
import torchaudio
import torch
import math 
import numpy as np
import torch.nn as nn
import torchaudio.transforms as T
import torchaudio.functional as F
import librosa
import random
from enum import Enum, auto
from fractions import Fraction
import matplotlib.pyplot as plt
import hashlib
import json
import h5py
import inspect

def hash_tensor(tensor: torch.Tensor) -> str:
    """
    Compute a SHA-256 hash of a tensor.
    
    Args:
        tensor (torch.Tensor): Input tensor.
    
    Returns:
        str: SHA-256 hash of the tensor.
    """
    tensor_bytes = tensor.numpy().tobytes()
    return hashlib.md5(tensor_bytes).hexdigest()


def hash_parameters(parameters: dict) -> str:
    """
    Compute a SHA-256 hash of a dictionary of parameters.
    
    Args:
        parameters (dict): Input dictionary of parameters.
    
    Returns:
        str: SHA-256 hash of the parameters.
    """
    parameters_bytes = json.dumps(parameters).encode()
    return hashlib.md5(parameters_bytes).hexdigest()


def hash_and_store_parameters(frame, waveform_array: torch.Tensor, verbose=True) -> str:
    # Get the arguments as a dictionary
    args, _, _, values = inspect.getargvalues(frame)
    assert 'waveform_array' in args
    args_dict = {arg: values[arg] for arg in args if arg != 'waveform_array'}
    waveform_hash = hash_tensor(waveform_array)
    args_dict['waveform_hash'] = waveform_hash

    existing_parameterizations_dir = './parameterizations'

    os.makedirs(existing_parameterizations_dir, exist_ok=True)
    # each folder in existing_parameterizations_dir will be named after the hash of the waveform_array
    subdir_for_waveform = os.path.join(existing_parameterizations_dir, waveform_hash)

    os.makedirs(subdir_for_waveform, exist_ok=True)
    # inside this dir, there shall be a json file that maps hashes of the parameterizations to the parameterizations
    parameterizations_fp = os.path.join(subdir_for_waveform, 'parameterizations.json')
    # get parameterizations hash
    params_hash = hash_parameters(args_dict)
    
    d = {params_hash: args_dict}
    if not os.path.exists(parameterizations_fp):
        if verbose:
            print(f'Writing parameterizations to new file {parameterizations_fp}')
        with open(parameterizations_fp, 'w') as f:
            json.dump(d, f, indent=4)
    else:
        if verbose:
            print(f'Updating parameterizations in file {parameterizations_fp}')
        with open(parameterizations_fp, 'r') as f:
            existing_d = json.load(f)
        existing_d.update(d)
        with open(parameterizations_fp, 'w') as f:
            json.dump(existing_d, f, indent=4)
    
    # the name of the data file to save will be <params_hash>.h5
    data_fp = os.path.join(subdir_for_waveform, f'{params_hash}.h5')
    return data_fp


def save_tensors_to_hdf5(stft, mfcc, pitch, resulting_data_fn, compression='gzip', compression_opts=9, verbose=True):
    """
    Save tensors to an HDF5 file with compression.
    
    Args:
        stft (torch.Tensor): STFT tensor.
        mfcc (torch.Tensor): MFCC tensor.
        pitch (torch.Tensor): Pitch tensor.
        resulting_data_fn (str): Path to the HDF5 file.
        compression (str): Compression algorithm to use ('gzip', 'lzf', 'szip').
        compression_opts (int): Compression level (0-9 for 'gzip').
    """
    if verbose:
        print(f'Saving tensors to {resulting_data_fn}')
    with h5py.File(resulting_data_fn, 'w') as f:
        f.create_dataset('stft', data=stft.numpy(), compression=compression, compression_opts=compression_opts, chunks=True)
        f.create_dataset('mfcc', data=mfcc.numpy(), compression=compression, compression_opts=compression_opts, chunks=True)
        f.create_dataset('pitch', data=pitch.numpy(), compression=compression, compression_opts=compression_opts, chunks=True)
    
def load_tensors_from_hdf5(resulting_data_fn, verbose=True):
    """
    Load tensors from an HDF5 file.
    
    Args:
        resulting_data_fn (str): Path to the HDF5 file.
    
    Returns:
        tuple: Tuple of STFT, MFCC, and pitch tensors.
    """
    if verbose:
        print(f'Loading tensors from {resulting_data_fn}')
    with h5py.File(resulting_data_fn, 'r') as f:
        stft = torch.tensor(f['stft'], dtype=torch.float32)
        mfcc = torch.tensor(f['mfcc'], dtype=torch.float32)
        pitch = torch.tensor(f['pitch'], dtype=torch.float32)
    return stft, mfcc, pitch

def set_seed(seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class WeightedMSELoss(nn.Module):
    def __init__(self):
        super(WeightedMSELoss, self).__init__()

    def forward(self, predictions: torch.tensor, targets: torch.tensor):
        # weigh each frequency bin by its index. the 0th bin should receive full weight, 
        # and each successive bin should receive .5 the weight of the previous bin.
        num_feats = predictions.shape[1]
        r = 0.45
        powvec = torch.arange(0, -r, -(r/num_feats), device=predictions.device)
        weights = torch.pow(2.0, 1.0*powvec)
        weights = (weights / torch.sum(weights)) * num_feats

        return torch.mean(weights * (predictions - targets) ** 2.0)

def get_criterion(s: str):
    if s == 'MSE':
        return torch.nn.MSELoss()
    elif s == 'WeightedMSE':
        return WeightedMSELoss()
    else:
        raise ValueError('Invalid criterion')

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
    elif os.path.isfile(audio_path)
        wave, fs = torchaudio.load(audio_path, normalize=True)
        audio_data = [process_wave(wave, fs, desired_sample_rate)]
    else:
        raise ValueError(f'Invalid audio path: {audio_path}')
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


def pitch_lin_to_log_scale(pitch: torch.Tensor, f_low: float, pitch_log_eps: float) -> torch.Tensor:
    assert f_low > 0
    pitch = torch.clip(pitch, min=f_low)
    return torch.log(pitch - f_low + pitch_log_eps)

def pitch_log_to_lin_scale(pitch: torch.Tensor, f_low: float, pitch_log_eps: float) -> torch.Tensor:
    return torch.exp(pitch) + f_low - pitch_log_eps


def closest_ratio(sample_rate_1, sample_rate_2, max_denominator=1000):
    # Calculate the exact ratio as a fraction
    ratio = sample_rate_1 / sample_rate_2
    if isinstance(ratio, torch.Tensor):
        ratio = ratio.item()
    # Find the closest fraction within the desired tolerance or max denominator
    closest_fraction = Fraction(ratio).limit_denominator(max_denominator)
    return closest_fraction.numerator, closest_fraction.denominator



def get_N_cycle_segments(waveform_array, sample_rate: float, window_size: int, hop_size: int, 
                   pitch_array,
                   voiced_probs=None,
                   cycles_per_window: int=1) -> tuple[list[np.ndarray], np.ndarray]:
    '''
    This calculates the N-cycle segments per window, by:
    -Based on correpsonding fundumanetal frequency, indexing the waveform to get the N-cycle segment
    -Resampling the audio for each window, such that the window contains exactly N cycles of the waveform at that point, where N is cycles_per_window.

    
    Args:
        waveform_array (np.ndarray): Audio waveform, in shape (1, num_samples)
        sample_rate (float): Sample rate of the audio
        window_size (int): Window size
        hop_size (int): Hop size
        pitch_array (np.ndarray): Pitch array, in Hz, in shape (num_windows,) 
        voided_probs (np.ndarray): Voiced probability array, in shape (num_windows,)
        cycles_per_window (int): Number of cycles per window

    Returns:
        segmented_waveforms (list[np.ndarray]): List of N-cycle segments, of length num_windows, where each np.ndarray is of shape (wavelength_given_f0_and_cycles_per_window,)
        resampled_waveform (np.ndarray): Resampled N-cycle wavelengths, in shape (num_windows, window_size)
    '''
    segmented_waveforms = []
    resampled_waveform = []

    target_wavelength =  window_size
    target_freq = sample_rate / target_wavelength


    total_frames = 1 + int(waveform_array.size(1) // hop_size)
    waveform_array = torch.nn.functional.pad(
        waveform_array,
        (window_size // 2, window_size // 2))
    
    # pre-compute the raw windows
    raw_frames = librosa.util.frame(waveform_array, frame_length=window_size, hop_length=hop_size)

    assert raw_frames.shape[-1] == total_frames

    if pitch_array.shape[0] == 1:
        pitch_array = pitch_array.squeeze()

    assert pitch_array.shape[0] == raw_frames.shape[-1]

    # resample each window to have N cycles of the waveform. 
    for i in range(pitch_array.shape[0]):
        f_0 = pitch_array[i]
        if voiced_probs is not None:
            voiced_prob = voiced_probs[i]
        else:
            voiced_prob = 1.0
        if f_0 == 0 or not voiced_prob:
            resampled_waveform.append(raw_frames[:, i])
            continue

        # we actually want to only pick out a single wavelength's worth of samples, centered in the middle of the window
        midpoint = window_size // 2
        wavelength_given_f0_and_cycles_per_window = (sample_rate / f_0) * cycles_per_window
        num_samples_in_wavelength = math.ceil(wavelength_given_f0_and_cycles_per_window)
        start_idx = midpoint - num_samples_in_wavelength // 2
        end_idx = start_idx + num_samples_in_wavelength
        x_i = raw_frames[0, start_idx:end_idx, i]
        segmented_waveforms.append(x_i)

        y_i = librosa.resample(x_i, 
                               orig_sr = target_freq * cycles_per_window, 
                               target_sr = f_0,
                               res_type='soxr_lq', 
                               scale=True, 
                               fix=True)
        assert len(y_i) >= window_size
        y_i = y_i[:window_size]
        resampled_waveform.append(y_i)

    return segmented_waveforms, np.stack(resampled_waveform)