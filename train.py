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
from save import save_rt_model
from torch.optim.lr_scheduler import ExponentialLR, StepLR, CyclicLR, MultiplicativeLR, MultiStepLR 
import matplotlib.animation as animation
from util import set_seed, get_criterion, get_encoded_layer_size, get_torch_device

import pacmap

def main():
    model_comment = ''
    existing_model_fp = ''

    set_seed(42)
    params = json.load(open('params.json'))

    assert params['n_fft'] >= params['window_size'], 'n_fft must be greater than or equal to window_size'

    X_train, Y_train, X_test, Y_test = get_data(audio_files_path=params['audio_files_path'], 
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

    device = get_torch_device()
    print(f'Using device: {device}')

    train_loader: torch.utils.data.DataLoader = torch.utils.data.DataLoader(list(zip(X_train, Y_train)), batch_size=params['batch_size'], shuffle=True)
    validation_loader: torch.utils.data.DataLoader = torch.utils.data.DataLoader(list(zip(X_test, Y_test)), batch_size=params['batch_size'], shuffle=True)


    net = network.GRUNet(X_train.shape[2],
                         hidden_size=params['hidden_size'], 
                         encoded_size=get_encoded_layer_size(params),
                         output_size=Y_train.shape[2], 
                         dropout_prob=params['dropout']).to(device)
    
    # pre-load saved weights/biases if available
    if os.path.exists(existing_model_fp):
        state_dict = torch.load(existing_model_fp)
        net.load_state_dict(state_dict)
        print(f'Loaded existing model from {existing_model_fp}')
    
    criterion = get_criterion(params['criterion'])

    if params['optimizer'] == 'Adam':
        optimizer = optim.Adam(net.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'], amsgrad=params['amsgrad'],)
    elif params['optimizer'] == 'AdamW':
        optimizer = optim.AdamW(net.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'], amsgrad=params['amsgrad'],)
    elif params['optimizer'] == 'SGD':
        optimizer = optim.SGD(net.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'], momentum=params['momentum'])
    
    scheduler = MultiplicativeLR(optimizer, lr_lambda=lambda epoch: params['lr_decay'])

    training_losses, validation_losses, weights = network.train(net, 
                                                       train_loader, 
                                                       criterion, 
                                                       optimizer, 
                                                       device, 
                                                       params['num_epochs'], 

                                                       validation_loader=validation_loader,
                                                       scheduler=scheduler,
                                                       record_weights_every=params['record_weights_every'],
                                                       print_every=20,
                                                       save_scratch_model_every=600,
                                                       scratch_model_dir='./scratch_model_dir',
                                                       num_batches=params['num_batches'])

    # name model file with date and time
    time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if True:    # save model in pytorch format
        model_fn = f'model_{time}{model_comment}.pth'
        # place in models folder
        model_fn = os.path.join('./models', model_fn)
        if not os.path.exists('last_network.json'):
            json.dump({'last_network': model_fn}, open('last_network.json', 'w'))
        else:
            last_network_tracker = json.load(open('last_network.json'))
            last_network_tracker['last_network'] = model_fn
            json.dump(last_network_tracker, open('last_network.json', 'w'))
        torch.save(net.state_dict(), model_fn)
        print(f'Model saved as {model_fn}')

    if True:    # rtneural file save
        model_fn = f'rt_model_{time}.json'
        # place in models folder
        model_fn = os.path.join('./rtneural_models', model_fn)
        if not os.path.exists('last_network.json'):
            json.dump({'last_rt_network': model_fn}, open('last_network.json', 'w'))
        else:
            last_network_tracker = json.load(open('last_network.json'))
            last_network_tracker['last_rt_network'] = model_fn
            json.dump(last_network_tracker, open('last_network.json', 'w'))
#**********************************************************************
        save_rt_model(net, model_fn)
#**********************************************************************
        print(f'Model saved as {model_fn}')

    if True:
        def plot_losses(losses, which_str: str):
            plt.figure()
            plt.plot(losses, label=f'{which_str} Loss')
            # display value of last loss at the point
            plt.text(len(losses)-1, losses[-1], f'{losses[-1]:.4f}')
            plt.yscale('log')
            plt.ylim([np.min(losses)-np.min(losses)*0.01, np.max(losses)+np.max(losses)*0.01])
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title(f'{which_str} Loss')
            plt.savefig(f'plots/{which_str}_losses_{time}.png')

        plot_losses(validation_losses, 'Validation')
        plot_losses(training_losses, 'Training')
        plt.show()

    if params['animate_weights']:
        weights = [w.cpu() for w in weights]
        animate_weights(weights)


def animate_weights(weights: np.array):
    images = [weights[i].detach().numpy() for i in range(len(weights))]
    num_frames = len(images)

    fig, ax = plt.subplots()
    # Initialize the plot with the first image
    im = ax.imshow(images[0], cmap='viridis', aspect='auto')
    #include colorbar
    plt.colorbar(im)

    # Update function for the animation
    def update(frame):
        im.set_array(images[frame])
        return [im]

    # Set the framerate (frames per second)
    fps = 20
    interval = 1000 / fps  # Interval in milliseconds

    # Create the animation
    ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=interval, blit=True, repeat=True)
    path = './plots/'
    time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = os.path.join(path, f'weights_animation_{time_str}.gif')
    ani.save(path, writer='pillow', fps=fps)

    # Display the animation
    plt.show()

if __name__ == '__main__':
    main()