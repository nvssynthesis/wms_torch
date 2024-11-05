import numpy as np 
import math 
# frequency estimator
from librosa import pyin
import matplotlib.pyplot as plt
import json
from typing import Optional


def sine(t, f_0: float):
    return np.sin(2 * np.pi * f_0 * t)

def tri(t, f_0: float, f_s: float):
    # additive triangle wave 
    # 8/pi^2 * sum_{k=0}^N (-1)^i * n^-2 * sin(2pi * f_0 * n * t), where
    # n = 2k + 1
    tri_wave = 0
    k=0
    while True:
        n = 2 * k + 1
        harmonic = f_0 * n
        valid_harmonic = np.where(harmonic <= f_s / 2, harmonic, np.nan)    # only consider harmonics that are less than Nyquist
        tri_wave += np.nan_to_num((-1)**k * n**-2 * sine(t, valid_harmonic))# add the harmonic to the wave, if it is valid
        k += 1
        if np.all(harmonic > f_s / 2):
            break
    return 8 / np.pi**2 * tri_wave

def square(t, f_0: float, f_s: float):
    # additive square wave
    # 4/pi * sum_{k=0}^N sin(2pi * (2k-1) * f_0 * t) / (2k-1)
    square_wave = 0
    k=1
    while True:
        n = 2 * k - 1
        harmonic = f_0 * n
        valid_harmonic = np.where(harmonic <= f_s / 2, harmonic, np.nan)    # only consider harmonics that are less than Nyquist
        square_wave += np.nan_to_num(sine(t, valid_harmonic) / n)           # add the harmonic to the wave, if it is valid
        k += 1
        if np.all(harmonic > f_s / 2):
            break
    return 4 / np.pi * square_wave

def sawtooth(t, f_0: float, f_s: float):
    # additive sawtooth wave
    # 2/pi * sum_{k=1}^N (-1)^k * sin(2pi * k * f_0 * t) / k
    sawtooth_wave = 0
    k=1
    while True:
        harmonic = f_0 * k
        valid_harmonic = np.where(harmonic <= f_s / 2, harmonic, np.nan)        # only consider harmonics that are less than Nyquist
        sawtooth_wave += np.nan_to_num((-1)**k * sine(t, valid_harmonic) / k)   # add the harmonic to the wave, if it is valid
        k += 1
        if np.all(harmonic > f_s / 2):
            break
    return 2 / np.pi * sawtooth_wave

def generate_exponential_f0(t, f_min, f_max, change_prob):
    """
    Generate an f_0 vector with exponentially distributed random frequencies at random times.
    
    Parameters:
    t (np.ndarray): Time vector.
    f_min (float): Minimum frequency.
    f_max (float): Maximum frequency.
    change_prob (float): Probability of frequency change at each time step.
    
    Returns:
    np.ndarray: f_0 vector with exponentially distributed random frequencies.
    """
    # Initialize the f_0 vector
    f_0 = np.zeros_like(t)
    
    # Set the initial frequency
    current_frequency = np.power(10, np.random.uniform(np.log10(f_min), np.log10(f_max)))
    f_0[0] = current_frequency
    
    # Iterate through the time vector and assign frequencies
    for i in range(1, len(t)):
        if np.random.rand() < change_prob:
            current_frequency = np.power(10, np.random.uniform(np.log10(f_min), np.log10(f_max)))
        f_0[i] = current_frequency
    
    return f_0

def main(sample_rate, f_low, f_high, 
         win_size, hop_size,
         duration_in_seconds=0.2,
         prob_of_freq_change=1e-3):
    t = np.linspace(0, duration_in_seconds, math.ceil(sample_rate * duration_in_seconds))
    sig = sawtooth(t, 
              f_0=generate_exponential_f0(t, f_low, f_high, prob_of_freq_change),
              f_s=sample_rate) 
    # detect the frequency
    f0, voiced_flag, voiced_probs = pyin(sig, fmin=f_low, fmax=f_high, sr=sample_rate,
                                         win_length=win_size, hop_length=hop_size,
                                         resolution=0.1,
                                         max_transition_rate=90, # in octaves per second
                                         center=True,
                                         pad_mode='constant',
                                         fill_na=None)

    f0_frames = np.arange(0, len(f0)) * hop_size / sample_rate


    fig, ax1 = plt.subplots()

    # Plot the signal on the primary y-axis
    ax1.plot(t, sig, 'b-', label='Signal')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Signal', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.set_ylim([-1, 1])

    # Create a secondary y-axis and plot the frequency
    ax2 = ax1.twinx()
    ax2.plot(f0_frames, f0, 'r.', label='Estimated Frequency')
    ax2.set_ylabel('Frequency (Hz)', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    ax2.set_yscale('log')  
    ax2.set_ylim([f_low/2, f_high*2])

    fig.tight_layout()  # Adjust layout to make room for both y-axes
    plt.title('Signal and Estimated Frequency')
    plt.show()

if __name__ == '__main__':
    params = json.load(open('params.json'))
    main(sample_rate=params['sample_rate'], 
         f_low=params['f_low'], 
         f_high=params['f_high'],
         win_size=params['window_size'],
         hop_size=params['hop_size'],)