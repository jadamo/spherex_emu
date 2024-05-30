import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

class block_resnet(nn.Module):
    
    def __init__(self, input_dim, output_dim, num_layers,
                 skip_connection):
        
        super().__init__()

        self.layers = nn.Sequential()
        self.layers.add_module("layer0",    nn.Linear(input_dim, output_dim))
        self.layers.add_module("ReLU", nn.ReLU())
        for i in range(num_layers-1):
            self.layers.add_module("layer"+str(i+1), nn.Linear(output_dim, output_dim))
            self.layers.add_module("bn"+str(i+1),    nn.BatchNorm1d(output_dim))
            self.layers.add_module("ReLU",      nn.ReLU())
    
        if skip_connection:
            self.skip_layer = nn.Linear(input_dim, output_dim)
            self.bn = nn.BatchNorm1d(output_dim)

    def forward(self, X):
        Y = self.layers(X)
        return Y + self.bn(self.skip_layer(X))