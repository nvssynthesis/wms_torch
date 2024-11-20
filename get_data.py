import util 
import features
import torch
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pacmap

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


def get_data(audio_files_path, sample_rate, window_size, hop_size, n_fft, fft_type, power, 
             n_mfcc=13, n_mel=23, 
             mfcc_dim_reduction=None,
             f_low=85, f_high=2000, 
             include_voicedness=True,
             pitch_detection_method='crepe',
             cycles_per_window=None,
             training_seq_length=20,
             require_sequential_data: bool=True,
            random_state=12345):
    audio_data = util.load_audio_files(audio_files_path, sample_rate)
    audio_tensor = util.concatenateWaveforms(audio_data, window_size)

    if fft_type == 'complex':
        raise NotImplementedError('Complex FFT training and inference not yet implemented')
    elif fft_type == 'real':
        pass
    else:
        raise ValueError(f'Unrecognized fft_type: {fft_type}')
    
    stft, mfcc, pitch = features.getFeatures(audio_tensor, sample_rate, n_fft, window_size, hop_size, 
                                power=power, n_mfcc=n_mfcc, n_mel=n_mel, 
                                center=True, 
                                f_low=f_low, f_high=f_high, 
                                include_voicedness=include_voicedness,
                                pitch_detection_method=pitch_detection_method,
                                cycles_per_window=cycles_per_window)

    if mfcc_dim_reduction is not None:
        if mfcc_dim_reduction == 'pacmap':
            mfcc, embedding = transform_via_pacmap(mfcc, 
                                        n_components=3, n_neighbors=10, 
                                        MN_ratio=0.5, FP_ratio=2.0, 
                                        distance='euclidean')
            mfcc = torch.tensor(mfcc, dtype=torch.float32)
        else:
            raise ValueError(f'Unrecognized mfcc_dim_reduction method: {mfcc_dim_reduction}')

    # input is mfcc and pitch
    # output is stft
    X = torch.cat((mfcc, pitch), dim=1) 
    Y = stft
    
    if require_sequential_data:
        # split the data into sequences
        # in order to be sure that X and Y get subsequenced in accordance with each other,
        # we need to create a single sequence from the X and Y data, and then split them
        # after the fact
        combined = torch.cat((X, Y), dim=1)
        combined = util.make_sequences(combined, sequence_length=training_seq_length)
        X = combined[:, :, :-Y.shape[1]]
        Y = combined[:, :, -Y.shape[1]:]


    # split the data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=random_state)

    return X_train, Y_train, X_test, Y_test
