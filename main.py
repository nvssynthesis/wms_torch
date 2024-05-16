'''
this script takes in an audio file (or a folder of audio files) and outputs a trained neural network model that maps timbral features to spectra. 
steps:
1. load audio files
2. window the audio files
3. extract timbral features (MFCCs) per window
4. extract spectral features (STFT) per window
5. train a neural network to map timbral features to spectral features
6. save the model
'''
import torch.utils
import torch.utils.data
import torch.utils.data.dataloader
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import network
from get_data import get_data
import numpy as np
import json 
import datetime
import os 

def main():
    params = json.load(open('params.json'))

    BATCH_SIZE = 32
    LEARNING_RATE = 0.002
    EPOCHS = 1

    X_train, Y_train, X_test, Y_test = get_data(audio_files_path=params['audio_files_path'], 
                                                sample_rate=params['sample_rate'], 
                                                window_size=params['window_size'], 
                                                hop_size=params['hop_size'],
                                                n_fft=params['n_fft'],
                                                power=params['power'],
                                                n_mfcc=params['n_mfcc'],
                                                n_mel=params['n_mel'],
                                                f_low=params['f_low'],
                                                f_high=params['f_high'])

    device = torch.device('cpu')
    print(f'Using device: {device}')

    train_loader: torch.utils.data.dataloader = torch.utils.data.DataLoader(list(zip(X_train, Y_train)), batch_size=BATCH_SIZE)

    net = network.Net(X_train.shape[1], Y_train.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    losses = network.train(net, train_loader, criterion, optimizer, device, EPOCHS)

    # name model file with date and time
    time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_fn = f'model_{time}.pth'
    # place in models folder
    model_fn = os.path.join('./models', model_fn)
    last_network_tracker = json.load(open('last_network.json'))
    last_network_tracker['last_network'] = model_fn
    json.dump(last_network_tracker, open('last_network.json', 'w'))
    torch.save(net.state_dict(), model_fn)
    print(f'Model saved as {model_fn}')

    plt.figure()
    plt.plot(losses)
    plt.ylim([0, np.max(losses)+0.1])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.savefig(f'losses_{time}.png')
    plt.show()

if __name__ == '__main__':
    main()