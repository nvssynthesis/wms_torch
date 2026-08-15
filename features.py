import inspect
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
import librosa
from scipy.fft import dct
import torch
import torchaudio.functional as F
import torchcrepe 
import matplotlib.pyplot as plt
import pacmap
from util import get_N_cycle_segments, pitch_lin_to_log_scale, \
    hash_and_store_parameters, save_tensors_to_pt, load_tensors_from_pt, \
    get_torch_device
from data_to_midi import write_data_to_midi

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


def extract_mfcc_batch(segmented_waveforms, sample_rate, n_fft, n_mel, n_mfcc):
    # Stack all segments into one array: (num_segments, n_fft)
    windowed = []
    window = np.hanning(segmented_waveforms[0].shape[0])
    for wf in segmented_waveforms:
        wf = wf * window if wf.shape[0] == window.shape[0] else wf * np.hanning(wf.shape[0])
        if wf.shape[0] < n_fft:
            wf = librosa.util.fix_length(wf, size=n_fft, mode='wrap')
        windowed.append(wf)
    windowed = np.stack(windowed)  # (num_segments, n_fft)

    # Compute power spectrum for ALL segments at once via FFT (vectorized over axis=1)
    spectrum = np.abs(np.fft.rfft(windowed, n=n_fft, axis=1)) ** 2  # (num_segments, n_fft//2 + 1)

    # Build mel filterbank ONCE
    mel_basis = librosa.filters.mel(sr=sample_rate, n_fft=n_fft, n_mels=n_mel)  # (n_mels, n_fft//2+1)
    mel_spec = spectrum @ mel_basis.T  # (num_segments, n_mels)

    # Log + DCT (type 2) ONCE, vectorized
    log_mel = librosa.power_to_db(mel_spec.T).T  # librosa expects (n_mels, frames), so transpose
    mfcc = dct(log_mel, type=2, axis=1, norm='ortho')[:, :n_mfcc]  # (num_segments, n_mfcc)

    return torch.tensor(mfcc, dtype=torch.float32)

def extract_stft_batch(resampled_wave_matrix, n_fft):
    # Stack all waveforms into a single 2D array first
    wave_matrix = np.stack(resampled_wave_matrix)  # (num_frames, wave_length)

    # Apply Hanning window (broadcasts across all rows if wave_length is constant)
    window = np.hanning(wave_matrix.shape[1])
    windowed = wave_matrix * window  # (num_frames, wave_length)

    # Vectorized FFT across all frames at once
    stft = np.abs(np.fft.rfft(windowed, n=n_fft, axis=1))  # (num_frames, n_fft//2 + 1)

    stft = torch.tensor(stft, dtype=torch.float32)
    stft = torch.log1p(stft)

    return stft

def getMidiNameFromFeaturesFile(feature_data_fn):
    midi_filename = feature_data_fn.replace('.pt', '.mid')
    # also need to move it from ./parameterizations to ./midi_files
    midi_filename = midi_filename.replace('parameterizations', 'midi_files')
    # make sure the directory exists
    os.makedirs(os.path.dirname(midi_filename), exist_ok=True)
    return midi_filename

def getAudioNameFromFeaturesFile(feature_data_fn):
    audio_filename = feature_data_fn.replace('.pt', '.wav')
    # also need to move it from ./parameterizations to ./audio_files
    audio_filename = audio_filename.replace('parameterizations', 'audio_files')
    os.makedirs(os.path.dirname(audio_filename), exist_ok=True)
    return audio_filename


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
                center=False,
                force_recompute_features=False,
                verbose=True): 
    frame = inspect.currentframe()  # TODO: this is not a good way to ensure identity of params. Instead we could use a map
    
    if verbose:
        print("Hashing, storing, and potentially loading pre-computed parameters...")
    resulting_data_fn = hash_and_store_parameters(frame, waveform_array)
    if os.path.exists(resulting_data_fn) and not force_recompute_features:
        print('Loading pre-computed features...')
        stft, mfcc, pitch = load_tensors_from_pt(resulting_data_fn)

        midi_fn = getMidiNameFromFeaturesFile(resulting_data_fn)
        if not os.path.exists(midi_fn):
            print('Writing feature data to MIDI...')
            write_data_to_midi(torch.cat((mfcc, pitch), dim=1), f_low, include_voicedness, hop_size, midi_fn)

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

    mfcc = extract_mfcc_batch(segmented_waveforms, sample_rate, n_fft, n_mel, n_mfcc)  # (num_segments, n_mfcc)
    

    # calculate fft for each resampled waveform
    if verbose:
        print('Calculating STFT...')
    stft = extract_stft_batch(resampled_wave_matrix, n_fft)  # (num_segments, n_fft//2 + 1)


    if normalize_mfcc:
        # normalize the mfcc
        scaler = StandardScaler()
        mfcc = scaler.fit_transform(mfcc)
        mfcc = torch.tensor(mfcc, dtype=torch.float32)


    if mfcc_dim_reduction == '':
        mfcc_dim_reduction = None 
    
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
        pitch = pitch_lin_to_log_scale(pitch, f_low)

    pitch = pitch.unsqueeze(1)

    # check that the shapes' last dimensions are the same
    assert mfcc.shape[0] == stft.shape[0] == pitch.shape[0]

    if include_voicedness:
        assert voicedness.shape[0] == pitch.shape[0]
        pitch = torch.cat((pitch, voicedness.unsqueeze(1)), dim=1)

    if verbose:
        print(f'Saving features (based on this parameterization (of this audio dataset)) to disk at {resulting_data_fn}')
    save_tensors_to_pt(stft, mfcc, pitch, resulting_data_fn)


    midi_fn = getMidiNameFromFeaturesFile(resulting_data_fn)
    print('Writing feature data to MIDI...')
    write_data_to_midi(torch.cat((mfcc, pitch), dim=1), f_low, include_voicedness, hop_size, midi_fn)

        
    return stft, mfcc, pitch