import torch
import torch.nn as nn
from torch.nn import functional as F

import spherex_emu.models.blocks as blocks
from spherex_emu.utils import get_parameter_ranges

class MLP_single_sample_multi_redshift(nn.Module):

    def __init__(self, config_dict):
        super().__init__()

        output_dim = config_dict["output_kbins"] * 2
        num_zbins = config_dict["num_zbins"]
        __, bounds = get_parameter_ranges(config_dict)
        bounds = torch.Tensor(bounds)
        self.register_buffer("bounds", bounds)

        # to keep each redshift bin part seperate, use group convolution
        self.h1 = blocks.linear_with_channels(config_dict["num_cosmo_params"] + config_dict["num_bias_params"],
                                              config_dict["mlp_dims"][0], num_zbins)

        self.mlp_blocks = nn.Sequential()
        for i in range(config_dict["num_mlp_blocks"]):
            self.mlp_blocks.add_module("ResNet"+str(i+1),
                    blocks.block_parallel_resnet(config_dict["mlp_dims"][i],
                                        config_dict["mlp_dims"][i+1],
                                        config_dict["num_block_layers"],
                                        num_zbins,
                                        config_dict["use_skip_connection"]))
        
        #self.out_ell1 = nn.Linear(config_dict["mlp_dims"][-1], config_dict["output_kbins"])
        #self.out_ell2 = nn.Linear(config_dict["mlp_dims"][-1], config_dict["output_kbins"])
        self.h2 = blocks.linear_with_channels(config_dict["mlp_dims"][-1], output_dim, num_zbins)

    def normalize(self, params):
        return (params - self.bounds[:,0]) / (self.bounds[:,1] - self.bounds[:,0])

    def forward(self, X):
        X = self.normalize(X)
        X = X.flatten(1, 2)#.permute(1, 0, 2)

        X = F.relu(self.h1(X))
        for block in self.mlp_blocks:
            X = F.relu(block(X))

        X = torch.sigmoid(self.h2(X))

        return X