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
    

def train_epoch(model, data_loader, loss_fn, optimizer, device) -> float:
    for inputs, targets in data_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        # calculate the loss
        predictions = model(inputs)
        loss = loss_fn(predictions, targets)

        # backpropagate the loss and update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Loss: {loss.item()}')
    return loss.item()

def train(model, data_loader, loss_fn, optimizer, device, num_epochs):
    model.train()
    losses = np.zeros(num_epochs)
    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        loss = train_epoch(model, data_loader, loss_fn, optimizer, device)
        losses[epoch] = loss
        print('-' * 10)
    print('Finished training')
    return losses
