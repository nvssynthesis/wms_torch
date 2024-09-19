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
from torch.optim.lr_scheduler import ExponentialLR, StepLR
import matplotlib.animation as animation


class WeightedMSELoss(nn.Module):
    def __init__(self):
        super(WeightedMSELoss, self).__init__()

    def forward(self, predictions: torch.tensor, targets: torch.tensor):
        # weigh each frequency bin by its index. the 0th bin should receive full weight, 
        # and each successive bin should receive .5 the weight of the previous bin.
        num_feats = predictions.shape[1]
        r = 0.45
        powvec = torch.arange(0, -r, -(r/num_feats), device=predictions.device)
        weights = torch.pow(2.0, 1.0*powvec)
        weights = (weights / torch.sum(weights)) * num_feats

        return torch.mean(weights * (predictions - targets) ** 2.0)

def main():
    params = json.load(open('params.json'))

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

    train_loader: torch.utils.data.DataLoader = torch.utils.data.DataLoader(list(zip(X_train, Y_train)), batch_size=params['batch_size'], shuffle=True)
    validation_loader: torch.utils.data.DataLoader = torch.utils.data.DataLoader(list(zip(X_test, Y_test)), batch_size=params['batch_size'], shuffle=True)

    # net = network.Net(X_train.shape[1], Y_train.shape[1]).to(device)
    net = network.RNNNet(X_train.shape[2],
                         hidden_size=params['hidden_size'], 
                         output_size=Y_train.shape[2], 
                         num_layers=params['num_layers']).to(device)
    
    criterion = WeightedMSELoss()
    optimizer = optim.Adam(net.parameters(), lr=params['learning_rate'], weight_decay=0.001)
    scheduler = ExponentialLR(optimizer, gamma=0.99)
    # scheduler2 = StepLR

    training_losses, validation_losses, weights = network.train(net, 
                                                       train_loader, 
                                                       criterion, 
                                                       optimizer, 
                                                       device, 
                                                       params['num_epochs'], 
                                                       validation_loader=validation_loader,
                                                       scheduler=scheduler,
                                                       record_weights_every=1)
    animate_weights(weights)

    # name model file with date and time
    time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if True:    # save model in pytorch format
        model_fn = f'model_{time}.pth'
        # place in models folder
        model_fn = os.path.join('./models', model_fn)
        last_network_tracker = json.load(open('last_network.json'))
        last_network_tracker['last_network'] = model_fn
        json.dump(last_network_tracker, open('last_network.json', 'w'))
        torch.save(net.state_dict(), model_fn)
        print(f'Model saved as {model_fn}')

    if True:    # rtneural file save
        model_fn = f'rt_model_{time}.json'
        # place in models folder
        model_fn = os.path.join('./rtneural_models', model_fn)
        last_network_tracker = json.load(open('last_network.json'))
        last_network_tracker['last_rt_network'] = model_fn
        json.dump(last_network_tracker, open('last_network.json', 'w'))
#**********************************************************************
        save_rt_model(net, model_fn)
#**********************************************************************
        print(f'Model saved as {model_fn}')

    if True:
        plt.figure()
        plt.plot(training_losses, label='Training Loss')
        plt.plot(validation_losses, label='Validation Loss')
        plt.ylim([0, np.max(training_losses)+np.max(training_losses)*0.1])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.savefig(f'plots/losses_{time}.png')
        plt.show()


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