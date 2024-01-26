import torch
import torch.nn as nn
import torch.optim as optim


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.input_layer = nn.Linear(input_size, hidden_sizes[0])
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_sizes[i], hidden_sizes[i+1]) for i in range(len(hidden_sizes)-1)])
        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)
        
        self.activation = nn.ModuleList([nn.Sigmoid(), nn.ReLU6()] + [nn.Sigmoid() for _ in range(len(hidden_sizes)-2)] + [nn.Identity()])

    def forward(self, x):
        x = torch.relu(self.input_layer(x))
        for layer, activation in zip(self.hidden_layers, self.activation):
            x = activation(layer(x))
        x = self.output_layer(x)
        return x
    


class Critic(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()
        
        self.input_layer = nn.Linear(input_size, hidden_sizes[0])
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_sizes[i], hidden_sizes[i+1]) for i in range(len(hidden_sizes)-1)])
        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)
        
        self.activation = nn.ModuleList([nn.Sigmoid(), nn.ReLU6()] + [nn.Sigmoid() for _ in range(len(hidden_sizes)-2)] + [nn.Identity()])

    def forward(self, x):
        x = torch.relu(self.input_layer(x))
        for layer, activation in zip(self.hidden_layers, self.activation):
            x = activation(layer(x))
        x = self.output_layer(x)
        return x