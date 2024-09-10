import torch
import torch.nn as nn
from torch.nn import functional as F

import spherex_emu.models.blocks as blocks
from spherex_emu.utils import get_parameter_ranges, load_config_file
from spherex_emu.filepaths import base_dir

class MLP_single_sample_single_redshift(nn.Module):

    def __init__(self, config_dict):
        super().__init__()

        output_dim = config_dict["num_kbins"] * 2
        cosmo_file = load_config_file(base_dir + config_dict["cosmo_dir"])
        __, bounds = get_parameter_ranges(cosmo_file)
        self.register_buffer("bounds", torch.Tensor(bounds.T))

        self.h1 = nn.Linear(config_dict["num_cosmo_params"] + config_dict["num_bias_params"],
                            config_dict["mlp_dims"][0])
        self.mlp_blocks = nn.Sequential()
        for i in range(config_dict["num_mlp_blocks"]):
            self.mlp_blocks.add_module("ResNet"+str(i+1),
                    blocks.block_resnet(config_dict["mlp_dims"][i],
                                        config_dict["mlp_dims"][i+1],
                                        config_dict["num_block_layers"],
                                        config_dict["use_skip_connection"]))
        
        #self.out_ell1 = nn.Linear(config_dict["mlp_dims"][-1], config_dict["num_kbins"])
        #self.out_ell2 = nn.Linear(config_dict["mlp_dims"][-1], config_dict["num_kbins"])
        self.h2 = nn.Linear(config_dict["mlp_dims"][-1], output_dim)

    # TODO: Find a better fix than requiring this function
    def set_normalizations(self, output_normalizations):
        print("This is a redundant function right now")

    def normalize(self, params):
        return (params - self.bounds[0]) / (self.bounds[1] - self.bounds[0])

    def forward(self, X):
        X = self.normalize(X)

        X = F.relu(self.h1(X))
        for block in self.mlp_blocks:
            X = F.relu(block(X))

        #X1 = self.out_ell1(X)
        #X2 = self.out_ell2(X)
        #X = torch.sigmoid(torch.cat((X1, X2), dim=1))
        X = torch.sigmoid(self.h2(X))

        return X

