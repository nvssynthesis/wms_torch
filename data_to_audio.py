# converts the STFT data to output audio 
# phase is kept as 0 (all real)

from scipy.signal import istft
import torch
import numpy as np
# to write wave files 
from scipy.io.wavfile import write


def data_to_audio(stft_data, n_fft, hop_size, window='hann'):
    """
    Convert STFT data to audio waveform using inverse STFT.

    Parameters:
    - stft_data: Tensor of shape (num_frames, n_fft//2 + 1) representing the STFT magnitude.
    - n_fft: Number of FFT points used in the original STFT.
    - hop_size: Hop size used in the original STFT.
    - window: Type of window function to use for ISTFT.

    Returns:
    - audio_waveform: 1D numpy array representing the reconstructed audio waveform.
    """
    
    # Convert tensor to numpy array
    stft_data_np = stft_data.numpy()
    
    # Create a Hann window
    win = np.hanning(n_fft)
    
    # Perform inverse STFT
    _, audio_waveform = istft(stft_data_np.T, nperseg=n_fft, noverlap=n_fft - hop_size, window=win)
    
    return audio_waveform