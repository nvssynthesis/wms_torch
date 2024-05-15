import torch
from network import Net
from util import load_audio_files, concatenateWaveforms
from get_data import get_data
import matplotlib.pyplot as plt
import json

def predict(model, input):
    model.eval()
    with torch.no_grad():
        predicted = model(input)
    return predicted

def main():
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
    model_fn = 'model_2024-05-15_00-41-19.pth'
    state_dict = torch.load(model_fn)
    network.load_state_dict(state_dict)

    idx = 90
    input, target = X_test[idx], Y_test[idx]

    predicted = predict(network, input)
    inv_pow = 1 / params['power']
    plt.figure()
    plt.plot(target**inv_pow, label='target')
    plt.plot(predicted**inv_pow, label='predicted')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()