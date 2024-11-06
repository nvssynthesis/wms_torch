import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
import torchaudio 
import torchaudio.functional as F
import matplotlib.pyplot as plt

def getFeatures(waveform_array: torch.Tensor, 
                sample_rate, n_fft, window_size, hop_size, 
                f_low=85, f_high=3500,
                power=2.0, n_mel=23, n_mfcc=13, 
                normalize_mfcc=True,
                pitch_log_scale=True,
                pitch_log_eps=0.0001,
                center=False):
    

    transform = torchaudio.transforms.MFCC(
        sample_rate=sample_rate,
        n_mfcc=n_mfcc,
        melkwargs={"n_fft": n_fft, "win_length": window_size, "hop_length": hop_size, 
                   "n_mels": n_mel, "center": center, "power": power},
    )
    mfcc: torch.Tensor = transform(waveform_array)
    
    if normalize_mfcc:
        # normalize the mfcc
        scaler = StandardScaler()
        mfcc = scaler.fit_transform(mfcc.T.squeeze()).T
        mfcc = torch.tensor(mfcc,dtype=torch.float32).unsqueeze(0)

    # get fundamental frequencies for each window
    median_filter_win_length = 7
    pitch: torch.Tensor = F.detect_pitch_frequency(waveform_array, 
        sample_rate=sample_rate, 
        frame_time=hop_size/sample_rate, 
        win_length=median_filter_win_length,   # not window size, but number of windows fro which to compute median?
        freq_low=f_low, freq_high=f_high
    )
    if pitch_log_scale:
        pitch = torch.clip(pitch, min=f_low)
        pitch = torch.log(pitch - f_low + pitch_log_eps)

    stf_transform = torchaudio.transforms.Spectrogram(
        n_fft=n_fft,
        win_length=window_size,
        hop_length=hop_size,
        power=power,
        center=center
    )
    stft: torch.Tensor = stf_transform(waveform_array)


    # get rid of median_filter_win_length//2+1 frames at the beginning and end of other tensors
    frames_to_remove = median_filter_win_length//2+1
    print(f'mfcc len: {mfcc.shape[-1]}')
    print(f'pitch len: {pitch.shape[-1]}')
    if pitch.shape[-1] % 2 == 1:
        if mfcc.shape[-1] % 2 == 1:
            mfcc = mfcc[:, :, frames_to_remove//2:-frames_to_remove//2]
            stft = stft[:, :, frames_to_remove//2:-frames_to_remove//2]
        else:
            mfcc = mfcc[:, :, frames_to_remove//2:-frames_to_remove//2+1]
            stft = stft[:, :, frames_to_remove//2:-frames_to_remove//2+1]
    else:
        mfcc = mfcc[:, :, frames_to_remove//2:-frames_to_remove//2+1]
        stft = stft[:, :, frames_to_remove//2:-frames_to_remove//2+1]

    # check that the shapes' last dimensions are the same
    assert mfcc.shape[-1] == stft.shape[-1] == pitch.shape[-1]

    return stft, pitch, mfcc