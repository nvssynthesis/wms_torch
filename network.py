import torch.nn as nn
import torch 
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import os 
import json
import datetime
from time import time
from save import save_rt_model

class GRUNet(nn.Module):
    def __init__(self, input_size, hidden_size, encoded_size, output_size, dropout_prob: float = 0.1):
        super(GRUNet, self).__init__()
        self.hidden_size = hidden_size
        self.input_encoder = nn.Sequential(
            nn.Linear(input_size, encoded_size),
            nn.ReLU(),
        )
        self.gru = nn.GRU(encoded_size, hidden_size, 1, batch_first=True, dropout=dropout_prob)
        self.dense_layers = nn.Sequential(
            nn.Linear(hidden_size, output_size),
            nn.ReLU(),
        )

    def forward(self, x, h0):
        # forward pass through input encoder
        out = self.input_encoder(x)
        # Forward pass through GRU
        out, hn = self.gru(out, h0)

        out = self.dense_layers(out)
        return out, hn

    def init_hidden(self, batch_size, init_method='zeros'):
        if init_method == 'zeros':
            return torch.zeros(1, batch_size, self.hidden_size)
        elif init_method == 'normal':
            return torch.randn(1, batch_size, self.hidden_size)
        else:
            raise ValueError('Invalid init method')
    
    def get_weights(self) -> list[torch.Tensor]:
        # gets all weights in the model, both in the gru and in the dense layers
        weights = []
        for name, param in self.named_parameters():
            weights.append(param)
        return weights
        

def train_epoch(model: GRUNet, data_loader: torch.utils.data.DataLoader, loss_fn, optimizer, device, 
                validation_loader: torch.utils.data.DataLoader = None,
                num_batches: int = None):
    model.train()
    epoch_loss = 0.0
    val_loss = 0.0
    
    n_training_samples = len(data_loader.dataset)
    n_validation_samples = len(validation_loader.dataset) if validation_loader else 1

    for batch_idx, (inputs, targets) in enumerate(data_loader):
        if num_batches is not None and batch_idx >= num_batches:
            break
        inputs = inputs.to(device)
        targets = targets.to(device)

        # subsequences chosen randomly, more likely to use short subsequences (which would use a higher subseq_limiting_idx)
        # subseq_limiting_idx = np.random.choice(np.arange(1, inputs.size(1)), p=np.arange(1, inputs.size(1)) / np.sum(np.arange(1, inputs.size(1))))
        # subseq_inputs = inputs[:, subseq_limiting_idx:, :]

        # subseq_inputs = subseq_inputs.to(device)
        h0 = model.init_hidden(inputs.size(0)).to(device)
        optimizer.zero_grad()
        outputs, hn = model(inputs, h0)
        loss = loss_fn(outputs, targets) # this is the last frame of the target, necessary for gru. for dense, it would be just targets.
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    if validation_loader:
        model.eval()
        with torch.no_grad():
            for val_batch_idx, (val_inputs, val_targets) in enumerate(validation_loader):
                if num_batches is not None and val_batch_idx >= num_batches:
                    break
                val_inputs = val_inputs.to(device)
                val_targets = val_targets.to(device)

                # val_subseq_inputs = val_inputs[:, subseq_limiting_idx:, :]
                # val_subseq_inputs = val_subseq_inputs.to(device)

                val_h0 = model.init_hidden(val_inputs.size(0)).to(device)
                val_outputs, val_hn = model(val_inputs, val_h0)
                val_loss += loss_fn(val_outputs, val_targets).item()
    
    return epoch_loss, (val_loss / n_validation_samples) * n_training_samples if validation_loader else 0.0


def train(model, data_loader, loss_fn, optimizer, device, num_epochs, validation_loader=None, scheduler=None, print_every=1, 
          record_weights_every=0,
          scratch_model_dir=None,
          save_scratch_model_every=15,
          num_batches=None):
    model.train()
    training_losses = np.zeros(num_epochs)
    validation_losses = np.zeros(num_epochs)
    weights = []

        
    for epoch in range(num_epochs):
        training_losses[epoch], validation_losses[epoch] = train_epoch(model, data_loader, loss_fn, optimizer, device, validation_loader, num_batches=num_batches)

        if epoch % print_every == 0:
            print(f'Epoch {epoch + 1}/{num_epochs} \n Training Loss: {training_losses[epoch]:.5f}')
            if validation_loader:
                print(f'Validation Loss: {validation_losses[epoch]:.5f}')
            if scheduler:
                print(f'Learning rate: {scheduler.get_last_lr()[0]}')
            for name, param in model.named_parameters():
                print(f'{name} has norm {torch.linalg.vector_norm(param)}')
            print('-' * 50)
        
        if record_weights_every and epoch % record_weights_every == 0:
            weights_curr = [we.flatten() for we in model.get_weights()]
            weights_cat = torch.cat(weights_curr)
            L = int(np.floor(np.sqrt(len(weights_cat))) ** 2)
            weights_cat = weights_cat[:L]
            L = int(np.sqrt(L))
            weights_shaped = weights_cat.reshape((L, L))
            weights.append(weights_shaped)
        
        #get norm of weights

        if scratch_model_dir and epoch % save_scratch_model_every == 0:      
            time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") 
            # create directory if it doesn't exist
            if not os.path.exists(scratch_model_dir):
                os.makedirs(scratch_model_dir)
    #**********************************************************************
            model_fn = f'model_{time_str}.json'
            model_fn = os.path.join(scratch_model_dir, model_fn)
            torch.save(model.state_dict(), model_fn)
    #**********************************************************************
            model_fn = f'rt_model_{time_str}.json'
            model_fn = os.path.join(scratch_model_dir, model_fn)
            save_rt_model(model, model_fn)
    #**********************************************************************
            print(f'Model saved as {model_fn}')


        if scheduler:
            scheduler.step()

    print('Finished training')
    return training_losses, validation_losses, weights
