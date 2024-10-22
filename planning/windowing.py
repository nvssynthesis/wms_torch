import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import numpy as np
import math

def process_phase_for_track(phase: float, track_idx: int, num_tracks: int):
    this_tracks_phase = phase - (track_idx / num_tracks)
    this_tracks_phase %= 1
    return this_tracks_phase

def hanning_osc(phase: float):
    phase = np.clip(phase, 0, 1)
    return 0.5 * (1 - np.cos(2 * np.pi * phase))

def new_color(color):
    # Assuming color is a tuple (h, s, v)
    return hsv_to_rgb(color)

def new_frequency(freq_min: float, freq_max: float):
    val = np.random.rand()
    # log scale
    return freq_min * (freq_max / freq_min) ** val

def waveform_transition_strategy_1(frequency_min: float = 0.001, frequency_max: float=0.5, change_freq_prob: float = 0.00001,
                                   overlap: float = 0.25, block_size: int = 512, total_length: int = 3000):
    num_tracks = math.ceil(1 / overlap)
    num_waveforms = math.ceil(1/overlap)
    tracks = np.zeros((num_tracks, num_waveforms, total_length))
    # current_track = 0
    # track just represents the fact that i want to visualize each window per cycle of overlaps as a different track
    # we are allowed to switch waveforms once per block, and the switch should occur the first time in the block that the window reaches its minimum (0 for hanning)
    current_waveform = 0
    current_phase = 0
    current_frequency = new_frequency(frequency_min, frequency_max)
    waveform_switch_allowed = False
    for block in range(0, total_length, block_size):
        waveform_switch_allowed = True
        for samp_idx in range(block, block + block_size):
            if samp_idx >= total_length:
                break
            for track_idx in range(len(tracks)):
                this_tracks_phase = process_phase_for_track(current_phase, track_idx, num_tracks)
                # this_tracks_phase = np.clip(this_tracks_phase, track_idx-0.25, track_idx + 1.25)
                tracks[track_idx, current_waveform, samp_idx] = hanning_osc(this_tracks_phase)
                # tracks[track_idx, current_waveform, samp_idx] = (this_tracks_phase)
            if np.random.rand() < change_freq_prob:
                current_frequency = new_frequency(frequency_min, frequency_max)
            current_phase += current_frequency
            if current_phase > 1:
                current_phase -= 1
                # current_track += 1
                # current_track %= num_tracks
                if waveform_switch_allowed:
                    current_waveform += 1
                    current_waveform %= num_waveforms
                    waveform_switch_allowed = False


    plt.figure(figsize=(15, 4))
    colors = [new_color((i / num_waveforms, 1.0, 1.0)) for i in range(num_waveforms)]
    offset_coeff = 2
    offset_hack = np.linspace(0, -0.01, total_length) # so that peaks begin being sensed at the start and we can then proceed from the previous peak to the end
    # plot each track with a different offset        
    for waveform_idx in range(num_waveforms):
        for track_idx in range(num_tracks):
            plt.plot(tracks[track_idx, waveform_idx, :] - track_idx * offset_coeff, color=colors[waveform_idx])

            this_track_and_wf = tracks[track_idx, waveform_idx, :]
            this_track_and_wf += offset_hack
            prev_peak_idx = 0
            # Find the peak value and its index
            peak_idx = np.argmax(this_track_and_wf)
            peak_value = this_track_and_wf[peak_idx]
            while peak_value > 0.989:
                peak_plot_pt = peak_value - track_idx * offset_coeff
                # Add a label at the peak
                if peak_idx - prev_peak_idx > 20:
                    plt.text(peak_idx, peak_plot_pt, f'Wave #{waveform_idx}', fontsize=8, ha='center', va='bottom')
                prev_peak_idx = peak_idx
                seq_to_inspect = this_track_and_wf[prev_peak_idx+1:]
                if not len(seq_to_inspect):
                    break
                peak_idx = np.argmax(seq_to_inspect) + prev_peak_idx + 1
                peak_value = this_track_and_wf[peak_idx]

    # make lines at the end of each block
    for i in range(0, total_length, block_size):
        plt.axvline(i, color='k', linestyle='--', alpha=0.5, linewidth=0.5)
    plt.show()
        


if __name__ == '__main__':
    waveform_transition_strategy_1(frequency_min=0.009, frequency_max=0.009, change_freq_prob=0.0013,
                                   overlap=0.3333, block_size=512, total_length=700)