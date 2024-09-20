import torch.nn as nn
import torch 
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
class Net(nn.Module):
    def __init__(self, input_size, output_size):
        super(Net, self).__init__()
        self.dense_layers = nn.Sequential(
            nn.Linear(input_size, 80), 
            nn.ReLU(),
            nn.Linear(80, 128), 
            nn.ReLU(),
            nn.Linear(128, output_size),
            nn.ReLU()
        ) 

    def forward(self, x):
        y = self.dense_layers(x)
        return y 

class RNNNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers: int = 1):
        super(RNNNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, nonlinearity='relu')
        self.dense_layers = nn.Sequential(
            nn.Linear(hidden_size, output_size),
            nn.ReLU(),
        )

    def forward(self, x, h0):
        # Forward pass through RNN
        out, hn = self.rnn(x, h0)
        # Use the hidden state of the last time step
        out = out[:, -1, :]
        # Flatten the output before passing to dense layers
        out = out.view(out.size(0), -1)
        out = self.dense_layers(out)
        return out, hn

    def init_hidden(self, batch_size):
        # Initialize hidden state with zeros
        return torch.zeros(self.num_layers, batch_size, self.hidden_size)
    
    def get_weights(self) -> list[torch.Tensor]:
        # gets all weights in the model, both in the rnn and in the dense layers
        weights = []
        for name, param in self.named_parameters():
            weights.append(param)
        return weights
        

def train_epoch(model, data_loader, loss_fn, optimizer, device, validation_loader=None):
    model.train()
    epoch_loss = 0.0
    val_loss = 0.0
    
    n_training_samples = len(data_loader.dataset)
    n_validation_samples = len(validation_loader.dataset) if validation_loader else 1
    for batch_idx, (inputs, targets) in enumerate(data_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Initialize hidden state
        h0 = model.init_hidden(inputs.size(0)).to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs, hn = model(inputs, h0)
        
        # Compute loss
        loss = loss_fn(outputs, targets[:, -1, :]) # this is the last frame of the target, necessary for rnn. for dense, it would be just targets.
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    if validation_loader:
        model.eval()
        with torch.no_grad():
            for val_inputs, val_targets in validation_loader:
                val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)
                
                # Initialize hidden state
                val_h0 = model.init_hidden(val_inputs.size(0)).to(device)
                
                # Forward pass
                val_outputs, val_hn = model(val_inputs, val_h0)
                
                # Compute loss
                val_loss += loss_fn(val_outputs, val_targets[:, -1, :]).item()
    
    return epoch_loss, (val_loss / n_validation_samples) * n_training_samples if validation_loader else 0.0


def train(model, data_loader, loss_fn, optimizer, device, num_epochs, validation_loader=None, scheduler=None, print_every=1, record_weights_every=0):
    model.train()
    training_losses = np.zeros(num_epochs)
    validation_losses = np.zeros(num_epochs)
    weights = []
    for epoch in range(num_epochs):
        training_losses[epoch], validation_losses[epoch] = train_epoch(model, data_loader, loss_fn, optimizer, device, validation_loader)

        if epoch % print_every == 0:
            print(f'Epoch {epoch + 1}/{num_epochs} \n Training Loss: {training_losses[epoch]:.5f}')
            if validation_loader:
                print(f'Validation Loss: {validation_losses[epoch]:.5f}')
            if scheduler:
                print(f'Learning rate: {scheduler.get_last_lr()[0]}')
            for name, param in model.named_parameters():
                print(f'{name} has norm {torch.norm(param)}')
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



        if scheduler:
            scheduler.step()

    print('Finished training')
    return training_losses, validation_losses, weights
