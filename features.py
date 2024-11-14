import numpy as np
from sklearn.preprocessing import StandardScaler
import librosa
import torch
import torchaudio 
import torchaudio.functional as F
import torchcrepe 
import matplotlib.pyplot as plt
import inspect
import os
import h5py
from util import get_N_cycle_segments, pitch_lin_to_log_scale, pitch_log_to_lin_scale, \
    hash_tensor, save_tensors_to_hdf5, hash_and_store_parameters, load_tensors_from_hdf5


def getFeatures(waveform_array: torch.Tensor, 
                sample_rate, n_fft, window_size, hop_size, 
                pitch_detection_method = 'crepe',
                include_voicedness=True,
                f_low=85, f_high=3500,
                cycles_per_window=None,
                power=2.0, n_mel=23, n_mfcc=13, 
                normalize_mfcc=True,
                pitch_log_scale=True,
                pitch_log_eps=0.0001,
                center=False):    
    # Get the current frame
    frame = inspect.currentframe()
    
    resulting_data_fn = hash_and_store_parameters(frame, waveform_array)
    if os.path.exists(resulting_data_fn):
        stft, mfcc, pitch = load_tensors_from_hdf5(resulting_data_fn)
        return stft, mfcc, pitch

    if pitch_detection_method == 'pyin':
        # pitch detect with librosa using librosa.pyin
        pitch, voiced_flag, voicedness = librosa.pyin(waveform_array.squeeze().numpy(), 
                             fmin=f_low, fmax=f_high, 
                             sr=sample_rate, 
                             frame_length=window_size, hop_length=hop_size, fill_na=None)
        pitch = torch.tensor(pitch, dtype=torch.float32)
        voicedness = torch.tensor(voicedness, dtype=torch.float32)

    elif pitch_detection_method == 'yin':
        if include_voicedness:
            raise ValueError('yin method does not support voicedness')
        # pitch detect with librosa using librosa.yin
        pitch = librosa.yin(waveform_array.squeeze().numpy(), 
                             fmin=f_low, fmax=f_high, 
                             sr=sample_rate, 
                             frame_length=window_size, hop_length=hop_size)
        pitch = torch.tensor(pitch, dtype=torch.float32)

    elif pitch_detection_method == 'crepe':
        assert int(hop_size) == hop_size
        if sample_rate != 16000:
            raise ValueError('crepe only supports sample rate of 16000. The workaround for this is not yet satisfactory.')
        # pitch detect with crepe
        pitch, voicedness = torchcrepe.predict(waveform_array, sample_rate=sample_rate, 
                                               hop_length=int(hop_size), 
                                               fmin=f_low, fmax=f_high, 
                                                model='tiny', decoder = torchcrepe.decode.viterbi, 
                                                return_periodicity = True, 
                                                batch_size=256, device='mps',
                                                pad=True)
        pitch = pitch.squeeze()
        voicedness = voicedness.squeeze()

    else:
        raise ValueError('pitch_detect_method not recognized')


    if cycles_per_window is not None:
        # then we must repitch the audio, for each window, to have N wavelengths perfectly fitting in the window, where N is cycles_per_window
        # this is done by resampling the audio for each window
        segmented_waveforms, resampled_wave_matrix = get_N_cycle_segments(waveform_array, sample_rate, window_size, hop_size, 
                                        pitch, voiced_probs=None, 
                                        cycles_per_window=cycles_per_window)
    else:
        raise NotImplementedError("Cycles per window is currently required")

    # calculate mfcc for each segmented waveform
    mfcc = []
    for wf in segmented_waveforms:
        # pad wf if necessary
        if wf.shape[0] < n_fft:
            wf = librosa.util.fix_length(wf, size=n_fft, mode='wrap')   # may want to use 'constant' instead of 'wrap'
        m = librosa.feature.mfcc(y=wf, 
                                 sr=sample_rate, 
                                 n_fft = n_fft, n_mfcc=n_mfcc, 
                                 dct_type=2, lifter=0, 
                                 hop_length=9999999)
        mfcc.append(torch.Tensor(m).squeeze())
    mfcc = torch.stack(mfcc)
    

    # calculate fft for each resampled waveform
    stft = []
    for i in range(len(resampled_wave_matrix)):
        stft.append(np.abs(np.fft.rfft(resampled_wave_matrix[i])))
    stft = torch.tensor(stft, dtype=torch.float32)

    if normalize_mfcc:
        # normalize the mfcc
        scaler = StandardScaler()
        mfcc = scaler.fit_transform(mfcc)
        mfcc = torch.tensor(mfcc, dtype=torch.float32)

    if pitch_log_scale:
        pitch = pitch_lin_to_log_scale(pitch, f_low, pitch_log_eps)

    pitch = pitch.unsqueeze(1)

    # check that the shapes' last dimensions are the same
    assert mfcc.shape[0] == stft.shape[0] == pitch.shape[0]

    if include_voicedness:
        assert voicedness.shape[0] == pitch.shape[0]
        pitch = torch.cat((pitch, voicedness.unsqueeze(1)), dim=1)

    # save the tensors to an HDF5 file
    save_tensors_to_hdf5(stft, mfcc, pitch, resulting_data_fn)

        
    return stft, mfcc, pitch