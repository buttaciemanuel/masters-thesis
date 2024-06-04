import torch
from torch import nn, optim
from torch.utils import data


class NeuralNetwork(nn.Module):
    '''
    Basic neural network with one hidden layer and ReLU activation. The
    combination of this model and the loss criterion constitutes a nonconvex
    loss objective. 
    '''

    def __init__(self, n_inputs: int, n_classes: int, n_hidden: int = 128):
        super().__init__()

        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(n_inputs, n_hidden)
        self.linear2 = nn.Linear(n_hidden, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = self.linear1(x)
        x = nn.functional.relu(x)
        return self.linear2(x)