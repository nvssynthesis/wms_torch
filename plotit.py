import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def plot_pacmap_embedding(X, embedding):
    emb_params = embedding.get_params()
    n_components = emb_params['n_components']
    n_neighbors = emb_params['n_neighbors']
    MN_ratio = emb_params['MN_ratio']
    FP_ratio = emb_params['FP_ratio']
    distance = emb_params['distance']
    # 3d scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # scatter using very small dots
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=0.1)
    # title
    plt.title('PaCMAP Embedding')
    # add miniature subtitle
    plt.suptitle(f'PaCMAP with {n_components} components, {n_neighbors} neighbors, MN ratio {MN_ratio}, FP ratio {FP_ratio} distance {distance}', fontsize=8)
    plt.show()


def plot_prediction(target, predicted, target_wave, predicted_wave, sample_idx, subsequence_idx):
    plt.figure()
    plt.clf()  
    # subplots with fft, waveforms
    plt.subplot(2, 1, 1)
    plt.plot(target, label='target')
    plt.plot(predicted, label='predicted')
    plt.legend()
    plt.title(f'Sample {sample_idx}, Subsequence 0-{subsequence_idx}')

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