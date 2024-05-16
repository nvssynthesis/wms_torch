import torch
from network import Net
from get_data import get_data
from plotit import plot_prediction
import json
import numpy as np 
from scipy.io.wavfile import write
from util import make_conjugate_symmetric

def predict(model, input):
    model.eval()
    with torch.no_grad():
        predicted = model(input)
    return predicted

def main():
    idx = 99
    device = torch.device('cpu')

    params = json.load(open('params.json'))

    _, _, X_test, Y_test = get_data(audio_files_path=params['audio_files_path'],
                                    sample_rate=params['sample_rate'],
                                    window_size=params['window_size'],
                                    hop_size=params['hop_size'],
                                    n_fft=params['n_fft'],
                                    power=params['power'],
                                    n_mfcc=params['n_mfcc'],
                                    n_mel=params['n_mel'],
                                    f_low=params['f_low'],
                                    f_high=params['f_high'])
    
    network = Net(X_test.shape[1], Y_test.shape[1]).to(device)
    last_network = json.load(open('last_network.json'))['last_network']
    state_dict = torch.load(last_network)
    network.load_state_dict(state_dict)

    input, target = X_test[idx], Y_test[idx]

    predicted = predict(network, input)
    inv_pow = 1 / params['power']
    target = target**inv_pow
    predicted = predicted**inv_pow

    # use inverse fft to convert the predicted and target to waveforms
    target_wave = torch.fft.ifft(make_conjugate_symmetric(torch.tensor(target, dtype=torch.complex64)))
    predicted_wave = torch.fft.ifft(make_conjugate_symmetric(torch.tensor(predicted, dtype=torch.complex64)))
    
    plot_prediction(target, predicted, target_wave, predicted_wave)

    # write the predicted audio to a file
    target_wave: np.ndarray = target_wave.numpy()
    predicted_wave = predicted_wave.numpy()
    scaled = np.int16(target_wave / np.max(np.abs(target_wave)) * 32767)
    write('./target.wav', 22050, scaled)
    scaled = np.int16(predicted_wave / np.max(np.abs(predicted_wave)) * 32767)
    write('./predicted.wav', 22050, scaled)

if __name__ == '__main__':
    main()