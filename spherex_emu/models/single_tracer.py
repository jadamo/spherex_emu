import torch
import torch.nn as nn
from torch.nn import functional as F

import spherex_emu.models.blocks as blocks
from spherex_emu.utils import organize_parameters

class MLP_single_tracer(nn.Module):

    def __init__(self, config_dict):
        super().__init__()

        output_dim = config_dict["output_kbins"] * 2
        __, bounds = organize_parameters(config_dict)
        bounds = torch.Tensor(bounds)
        self.register_buffer("bounds", bounds)

        self.h1 = nn.Linear(config_dict["num_cosmo_params"] + config_dict["num_bias_params"],
                            config_dict["mlp_dims"][0])
        self.mlp_blocks = nn.Sequential()
        for i in range(config_dict["num_mlp_blocks"]):
            self.mlp_blocks.add_module("ResNet"+str(i+1),
                    blocks.block_resnet(config_dict["mlp_dims"][i],
                                        config_dict["mlp_dims"][i+1],
                                        config_dict["num_block_layers"],
                                        config_dict["use_skip_connection"]))
        self.h2 = nn.Linear(config_dict["mlp_dims"][-1], output_dim)

    def normalize(self, params):
        return (params - self.bounds[:,0]) / (self.bounds[:,1] - self.bounds[:,0])

    def forward(self, X):
        X = self.normalize(X)

        X = F.leaky_relu(self.h1(X))
        for block in self.mlp_blocks:
            X = F.leaky_relu(block(X))
        X = F.leaky_relu(self.h2(X))
        return X

