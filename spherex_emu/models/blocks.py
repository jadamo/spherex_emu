import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

class block_resnet(nn.Module):
    
    def __init__(self, input_dim, output_dim, num_layers,
                 skip_connection):
        
        super().__init__()

        self.layers = nn.Sequential()
        self.layers.add_module(nn.Linear(input_dim, output_dim), nn.ReLu())
        for i in range(num_layers-1):
                self.layers.add_module(nn.Linear(output_dim, output_dim),
                                       nn.Batchnorm1d(output_dim),
                                       nn.ReLu())
    
        if skip_connection:
            self.skip_layer = nn.Linear(input_dim, output_dim)
            self.bn = nn.Batchnorm1d(output_dim)

    def forward(self, X):
        Y = self.layers(X)
        return Y + self.bn(self.skip_layer(X))