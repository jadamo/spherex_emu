import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

#https://github.com/pytorch/pytorch/issues/36591
class linear_with_channels(nn.Module):

    def __init__(self, input_dim, output_dim, num_channels):
        super().__init__()
        
        self.w = nn.Parameter(torch.zeros(num_channels, input_dim, output_dim))
        # with torch.no_grad():
        #     self.w[1,:,:] = torch.ones(input_dim, output_dim)
        self.b = nn.Parameter(torch.zeros(num_channels, 1, output_dim))

    def initialize_params(self, weight_initialization):
        if weight_initialization == "He":
            torch.nn.init.kaiming_uniform_(self.w)
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.w)
            bound = 1. / np.sqrt(fan_in)
            torch.nn.init.uniform_(self.b, -bound, bound)
        elif weight_initialization == "normal":
            nn.init.normal_(self.w, mean=0., std=0.1)
            nn.init.zeros_(self.b)
        elif weight_initialization == "xavier":
            nn.init.xavier_normal_(self.w)
        else: # if scheme is invalid, use normal initialization as a substitute
            nn.init.normal_(self.w, mean=0., std=0.1)
            nn.init.zeros_(self.w)

    def forward(self, X):
        
        # [b, c, in] x [b, in, out] = [b, c, out]
        # [c, b, in] x [c, in, out] = [c, b, out]
        X = X.permute(1, 0, 2) # <- [batch, channels, in] -> [channels, batch, in]
        X = torch.bmm(X, self.w) + self.b
        X = X.permute(1, 0, 2) # <- [channels, batch, out] -> [batch, channels, out]
        return X

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

class block_parallel_resnet(nn.Module):
    
    def __init__(self, input_dim, output_dim, num_layers, num_channels, skip_connection):
        
        super().__init__()

        self.layers = nn.Sequential()
        self.layers.add_module("layer0",  linear_with_channels(input_dim, output_dim, num_channels))
        self.layers.add_module("ReLU",    nn.ReLU())
        for i in range(num_layers-1):
            self.layers.add_module("layer"+str(i+1), linear_with_channels(output_dim, output_dim, num_channels))
            self.layers.add_module("bn"+str(i+1),    nn.BatchNorm1d(num_channels))
            self.layers.add_module("ReLU",           nn.ReLU())
    
        if skip_connection:
            self.skip_layer = linear_with_channels(input_dim, output_dim, num_channels)
            self.bn = nn.BatchNorm1d(num_channels)

    def forward(self, X):
        Y = self.layers(X)
        return Y + self.bn(self.skip_layer(X))