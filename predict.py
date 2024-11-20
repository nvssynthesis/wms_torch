import torch
from network import Net, GRUNet
from get_data import get_data
from plotit import plot_prediction
import json
import numpy as np 
from scipy.io.wavfile import write
from util import make_conjugate_symmetric
import datetime
import os 
import matplotlib.pyplot as plt

def predict(model: GRUNet, input):
    model.eval()
    with torch.no_grad():
        h0 = model.init_hidden(1)
        predicted = model(input.unsqueeze(0), h0)[0]
    return predicted

def make_name(base_name: str, ext: str, dir: str):
    time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    fn = f'{time}_{base_name}.{ext}'
    return os.path.join(dir, fn)

def write_audio(waveform, out_fs: int,  base_name: str, num_reps=1, dir: str = 'written_audio'):
    target_wave: np.ndarray = waveform.numpy()
    target_wave = np.tile(target_wave, num_reps)
    scaled = np.int16(target_wave / np.max(np.abs(target_wave)) * 32767)

    fn = make_name(base_name, 'wav', dir)
    write(fn, out_fs, scaled)

def main():
    indices = np.arange(0, 15)
    device = torch.device('cpu')
    params = json.load(open('params.json'))

    _, _, X_test, Y_test = get_data(audio_files_path=params['audio_files_path'], 
                                    sample_rate=params['sample_rate'], 
                                    window_size=params['window_size'], 
                                    hop_size=params['hop_size'],
                                    n_fft=params['n_fft'],
                                    fft_type=params['fft_type'],
                                    power=params['power'],
                                    n_mfcc=params['n_mfcc'],
                                    mfcc_dim_reduction=params['mfcc_dim_reduction'],
                                    n_mel=params['n_mel'],
                                    f_low=params['f_low'],
                                    f_high=params['f_high'],
                                    include_voicedness=params['include_voicedness'],
                                    pitch_detection_method=params['pitch_detection_method'],
                                    cycles_per_window=params['cycles_per_window'],
                                    training_seq_length=params['training_seq_length'],)
    
    # network = Net(X_test.shape[1], Y_test.shape[1]).to(device)
    network = GRUNet(X_test.shape[2], hidden_size=params['hidden_size'], output_size=Y_test.shape[2], num_layers=params['num_hidden_layers']).to(device)
    last_network = json.load(open('last_network.json'))['last_network']
    state_dict = torch.load(last_network)
    network.load_state_dict(state_dict)

    for idx in indices:
        input, target = X_test[idx], Y_test[idx]

        # target = target[-1, :]
        for i in range(1, len(input)):
            curr_inp = input[:i+1]
            predicted = predict(network, curr_inp)
            inv_pow = 1 / params['power']
            curr_target = target[i]
            curr_target = curr_target**inv_pow
            predicted = predicted**inv_pow

            curr_target = make_conjugate_symmetric(torch.tensor(curr_target, dtype=torch.complex64))
            predicted = make_conjugate_symmetric(torch.tensor(predicted.squeeze(), dtype=torch.complex64))

            # use inverse fft to convert the predicted and target to waveforms
            curr_target_wave = torch.fft.ifft(curr_target)
            predicted_wave = torch.fft.ifft(predicted)
            
            if params['do_plotting_on_prediction']:
                plot_prediction(curr_target, predicted, curr_target_wave, predicted_wave, idx, i)
                
            if params['write_audio']:
                print('Writing audio files...')
                write_audio(curr_target_wave, params['sample_rate'], f'target_{idx}')
                write_audio(predicted_wave, params['sample_rate'], f'predicted_{idx}')

            if params['write_spectrum']:
                print(f'Writing spectrum files...')
                # write the target and predicted spectra to .txt file
                target_fn = make_name(f'target_spectrum_{idx}', 'txt', 'written_spectra')
                predicted_fn = make_name(f'predicted_spectrum_{idx}', 'txt', 'written_spectra')
                if not os.path.exists('written_spectra'):
                    os.makedirs('written_spectra')
                np.savetxt(target_fn, target)
                np.savetxt(predicted_fn, predicted)

    print('Done!')

if __name__ == '__main__':
    main()