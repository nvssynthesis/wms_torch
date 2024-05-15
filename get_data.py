import util 
import features
import torch
from sklearn.model_selection import train_test_split

def get_data(audio_files_path, sample_rate, window_size, hop_size, n_fft, power, n_mfcc=13, n_mel=23, f_low=85, f_high=4000, random_state=12345):
    audio_data = util.load_audio_files(audio_files_path, sample_rate)
    audio_tensor = util.concatenateWaveforms(audio_data, window_size)

    stft, pitch, mfcc = features.getFeatures(audio_tensor, sample_rate, n_fft, window_size, hop_size, 
                                power=power, n_mfcc=n_mfcc, n_mel=n_mel, 
                                center=True, 
                                f_low=f_low, f_high=f_high)
    pitch /= f_high
    # input vector is mfcc and pitch
    # output vector is stft
    X = torch.cat((mfcc[0].T, pitch.T), dim=1) 
    Y = stft[0].T

    # normalize the data
    # since it's magnitude spectra, i don't want to change the mean, just the std
    X = X / X.std()
    Y = Y / Y.std()

    # split the data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=random_state)

    return X_train, Y_train, X_test, Y_test
