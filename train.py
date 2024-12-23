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
import os 
import datetime
import json 
import torch.utils
import torch.utils.data
import torch.utils.data.dataloader
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR, StepLR, CyclicLR, MultiplicativeLR, MultiStepLR 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import network
from get_data import get_data
from save import save_rt_model
from util import set_seed, get_criterion, get_encoded_layer_size, get_torch_device


def main(model_comment = None, existing_model_fp = None) -> None:
    '''
    Trains a neural network to map timbral features to spectral features.
    The actual bulk of the parameterization is done in the params.json file.
    The training takes several measures to save time and increase convenience, such as 
    -saving the abstracted features in a .pt file (so they only have to be computed if you change the parameters and/or the audio data),
    -saving the model in both pytorch and rtneural formats,
    -saving the model every 'save_scratch_model_every' epochs,
    -saving the names of the last-trained model in last_network.json, and
    -saving the training and validation losses in a plot png files

    Args:
    model_comment: str, optional
        A comment to add to the model filename, as well as e.g. plot filenames. It 
        can help to differentiate between different models or parameterizations.
    existing_model_fp: str, optional
        If a file path is provided, the model will be loaded from this file and
        training will continue from this point. Useful for resuming training if any
        issues arise or the model is still underfitting after num_epochs epochs.

    Returns:
        None
    '''
    if model_comment is None:
        model_comment = ''

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
    if existing_model_fp is not None and os.path.exists(existing_model_fp):
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

    training_losses, validation_losses, weights = network.train(model=net,
                                                       data_loader=train_loader, 
                                                       loss_fn=criterion, 
                                                       optimizer=optimizer, 
                                                       device=device, 
                                                       num_epochs=params['num_epochs'], 
                                                       validation_loader=validation_loader,
                                                       scheduler=scheduler,
                                                       record_weights_every=params['record_weights_every'],
                                                       print_every=20,
                                                       save_scratch_model_every=params['save_scratch_model_every'],
                                                       scratch_model_dir='./scratch_model_dir',
                                                       num_batches=params['num_batches'])

    # name model file with date and time
    time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if True:
        # save model in pytorch format
        save_model(net, time, rtneural=False, model_comment=model_comment)
        # save model in rtneural format
        save_model(net, time, rtneural=True, model_comment=model_comment)

    if True:
        plot_losses(validation_losses, 'Validation', time, model_comment)
        plot_losses(training_losses, 'Training', time, model_comment)
        plt.show()

    if params['animate_weights']:
        weights = [w.cpu() for w in weights]
        animate_weights(weights)

def save_model(net, time_str: str, rtneural: bool, model_comment: str=None, verbose: bool = True):
    if model_comment is None:
        model_comment = ''
    if not os.path.exists('./models'):
        os.makedirs('./models')
    if not os.path.exists('./rtneural_models'):
        os.makedirs('./rtneural_models')
    model_fn = f'model_{time_str}{model_comment}.pth' if not rtneural else f'rt_model_{time_str}.json'
    model_fn = os.path.join('./models', model_fn) if not rtneural else os.path.join('./rtneural_models', model_fn)
    if not os.path.exists('last_network.json'):
        key = 'last_network' if not rtneural else 'last_rt_network'
        json.dump({key: model_fn}, open('last_network.json', 'w'))
    else:
        last_network_tracker = json.load(open('last_network.json'))
        key = 'last_network' if not rtneural else 'last_rt_network'
        last_network_tracker[key] = model_fn
        json.dump(last_network_tracker, open('last_network.json', 'w'))
    if rtneural:
        save_rt_model(net, model_fn)
    else:
        torch.save(net.state_dict(), model_fn)
    if verbose:
        print(f'Model saved as {model_fn}')
        

def plot_losses(losses, which_str: str, time_str: str, model_comment: str=None):
    if model_comment is None:
        model_comment = ''
    plt.figure()
    plt.plot(losses, label=f'{which_str} Loss')
    # display value of last loss at the point
    plt.text(len(losses)-1, losses[-1], f'{losses[-1]:.4f}')
    plt.yscale('log')
    plt.ylim([np.min(losses)-np.min(losses)*0.01, np.max(losses)+np.max(losses)*0.01])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{which_str} Loss')
    if not os.path.exists('plots'):
        os.makedirs('plots')
    fn = f'plots/{which_str}_losses_{time_str}'
    if model_comment:
        fn += f'_{model_comment}'
    fn += '.png'
    plt.savefig(fn)


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