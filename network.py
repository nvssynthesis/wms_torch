import torch.nn as nn
import torch 
import numpy as np

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


def train_epoch(model, train_loader, loss_fn, optimizer, device, validation_loader=None) -> float:
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        # calculate the loss
        predictions = model(inputs)
        loss = loss_fn(predictions, targets)

        # backpropagate the loss and update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # get the validation loss
    if validation_loader is not None:
        model.eval()
        with torch.no_grad():
            for inputs, targets in validation_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                predictions = model(inputs)
                val_loss = loss_fn(predictions, targets)

    print(f'Loss: {loss.item()}')
    return loss.item(), val_loss.item()

def train(model, data_loader, loss_fn, optimizer, device, num_epochs, validation_loader=None):
    model.train()
    training_losses = np.zeros(num_epochs)
    validation_losses = np.zeros(num_epochs)
    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        training_losses[epoch], validation_losses[epoch] = train_epoch(model, data_loader, loss_fn, optimizer, device, validation_loader)
        print('-' * 10)
    print('Finished training')
    return training_losses, validation_losses
