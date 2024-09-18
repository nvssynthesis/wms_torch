import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def plot_prediction(target, predicted, target_wave, predicted_wave, sample_idx, subsequence_idx):
    plt.figure()
    plt.title(f'Sample {sample_idx}, Subsequence 0-{subsequence_idx}')
    # subplots with fft, waveforms
    plt.subplot(2, 1, 1)
    plt.plot(target, label='target')
    plt.plot(predicted, label='predicted')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(target_wave, label='target')
    plt.plot(predicted_wave, label='predicted')
    plt.legend()
    plt.show()

    return target_wave, predicted_wave

def plot_features(stft, mfccs, f0, voiced_prob=None):
    # plot matrix of windows using sns
    plt.figure(figsize=(10, 4))
    plt.subplot(4, 1, 1)
    sns.heatmap(np.log(stft[:, 0:-1:50]), cmap='viridis', cbar=False)
    plt.gca().invert_yaxis()  # Reverse the y-axis
    plt.title('STFT')

    plt.subplot(4, 1, 2)
    sns.heatmap(mfccs[:, 0:-1:50], cmap='viridis', cbar=False)
    plt.title('MFCCs')
    plt.gca().invert_yaxis()  # Reverse the y-axis

    # plot f0 in subplot
    plt.subplot(4, 1, 3)
    plt.plot(f0[0:-1:50], label='F0')
    plt.ylabel('F0')
    if voiced_prob is not None:
        ax2 = plt.gca().twinx()
        ax2.plot(voiced_prob[0:-1:50], 'y--', label='Voiced Probability')
        ax2.set_ylabel('Voiced Probability', color='y')
        plt.title('F0 and Voiced Probability')
    else:
        plt.title('F0')

    if voiced_prob is not None:
        plt.subplot(4,1,4)
        sns.scatterplot(x=f0, y=voiced_prob, alpha=0.5)

    plt.subplots_adjust(hspace=1.9)
    plt.show()