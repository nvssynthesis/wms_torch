import inspect
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
import librosa
import torch
import torchaudio.functional as F
import torchcrepe 
import matplotlib.pyplot as plt
import pacmap
from util import get_N_cycle_segments, pitch_lin_to_log_scale, \
    hash_and_store_parameters, save_tensors_to_pt, load_tensors_from_pt, \
    get_torch_device

def transform_via_pacmap(X, n_components=3, n_neighbors=5, MN_ratio=0.5, FP_ratio=0.5, 
                         distance='euclidean',
                         verbose=True) :
    distance = 'euclidean'
    embedding = pacmap.PaCMAP(n_components=n_components, n_neighbors=n_neighbors, 
                              MN_ratio=MN_ratio, FP_ratio=FP_ratio,
                              distance=distance)
    if verbose:
        print('Fitting PaCMAP...')
    X_embedded = embedding.fit_transform(X)
    return X_embedded, embedding

def getFeatures(waveform_array: torch.Tensor, 
                sample_rate, n_fft, window_size, hop_size, 
                pitch_detection_method = 'crepe',
                include_voicedness=True,
                f_low=85, f_high=3500,
                cycles_per_window=None,
                power=2.0, n_mel=23, n_mfcc=13, 
                mfcc_dim_reduction=None,
                normalize_mfcc=True,
                pitch_log_scale=True,
                pitch_log_eps=0.0001,
                center=False,
                verbose=True):    
    # Get the current frame
    frame = inspect.currentframe()
    
    if verbose:
        print("Hashing, storing, and potentially loading pre-computed parameters...")
    resulting_data_fn = hash_and_store_parameters(frame, waveform_array)
    if os.path.exists(resulting_data_fn):
        print('Loading pre-computed features...')
        stft, mfcc, pitch = load_tensors_from_pt(resulting_data_fn)
        return stft, mfcc, pitch


    if verbose:
        print('Extracting features from audio...')
        print(f'Detecting pitch via {pitch_detection_method}')
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
                                                batch_size=256, device=get_torch_device(),
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
                                        cycles_per_window=cycles_per_window,
                                        verbose=verbose)
    else:
        raise NotImplementedError("Cycles per window is currently required")

    # calculate mfcc for each segmented waveform
    if verbose: 
        print('Calculating MFCC...')
    mfcc = []
    for wf in segmented_waveforms:
        # window the waveform with hanning window
        wf = wf * np.hanning(wf.shape[0])

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
    if verbose:
        print('Calculating STFT...')
    stft = []
    for i in range(len(resampled_wave_matrix)):
        # hanning window the waveform
        x = resampled_wave_matrix[i]
        y = x * np.hanning(x.shape[0])
        stft.append(np.abs(np.fft.rfft(y, n=n_fft)))
    stft = torch.tensor(stft, dtype=torch.float32)

    if normalize_mfcc:
        # normalize the mfcc
        scaler = StandardScaler()
        mfcc = scaler.fit_transform(mfcc)
        mfcc = torch.tensor(mfcc, dtype=torch.float32)


    if mfcc_dim_reduction is not None:
        if mfcc_dim_reduction == 'pacmap':
            mfcc, embedding = transform_via_pacmap(mfcc, 
                                        n_components=3, n_neighbors=10, 
                                        MN_ratio=0.5, FP_ratio=2.0, 
                                        distance='euclidean',
                                        verbose=verbose)
            mfcc = torch.tensor(mfcc, dtype=torch.float32)
        else:
            raise ValueError(f'Unrecognized mfcc_dim_reduction method: {mfcc_dim_reduction}')


    if pitch_log_scale:
        pitch = pitch_lin_to_log_scale(pitch, f_low, pitch_log_eps)

    pitch = pitch.unsqueeze(1)

    # check that the shapes' last dimensions are the same
    assert mfcc.shape[0] == stft.shape[0] == pitch.shape[0]

    if include_voicedness:
        assert voicedness.shape[0] == pitch.shape[0]
        pitch = torch.cat((pitch, voicedness.unsqueeze(1)), dim=1)

    if verbose:
        print(f'Saving features (based on this parameterization (of this audio dataset)) to disk at {resulting_data_fn}')
    save_tensors_to_pt(stft, mfcc, pitch, resulting_data_fn)
        
    return stft, mfcc, pitch