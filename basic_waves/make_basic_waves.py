import numpy as np 
import math 
import json
import soundfile 
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from basic_waves.util import generate_wave, generate_exponential_f0, plot_signal_and_freq

'''
This script generates a synthetic waveform with a random frequency and waveform change. Its output can
be used to train the network with a dataset that has predictable results.
'''

def main(plot=False):
    params = json.load(open('params.json'))

    fs = params['sample_rate']
    dur_in_seconds = 100.0
    f_low = params['f_low']
    f_high = params['f_high']
    t = np.linspace(0.0, dur_in_seconds, math.ceil(fs * dur_in_seconds))
    sig = 0.707 * generate_wave(t, 
                        f_0=generate_exponential_f0(t, f_low, f_high, change_prob=1e-4),
                        f_s=fs,
                        prob_of_wave_change=3e-4)
    # write signal to wav file
    soundfile.write('./basic_waves/basic_waves.wav', sig, fs, subtype='float')
    if plot:
        plot_signal_and_freq(t, sig, fs, f_low, f_high, params['window_size'], params['hop_size'])

if __name__ == '__main__':
    main()
