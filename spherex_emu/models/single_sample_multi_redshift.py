import torch
import torch.nn as nn
from torch.nn import functional as F
import itertools

import spherex_emu.models.blocks as blocks
from spherex_emu.utils import get_parameter_ranges, load_config_file
from spherex_emu.filepaths import base_dir

class MLP_single_sample_multi_redshift(nn.Module):

    def __init__(self, config_dict):
        super().__init__()

        output_dim = config_dict["output_kbins"] * 2
        self.num_zbins = config_dict["num_zbins"]
        self.num_cosmo_params = config_dict["num_cosmo_params"]
        self.num_bias_params  = config_dict["num_bias_params"]
        
        cosmo_file = load_config_file(base_dir + config_dict["cosmo_dir"])
        __, bounds = get_parameter_ranges(cosmo_file)
        self.register_buffer("bounds", torch.Tensor(bounds.T))

        # to keep each redshift bin part seperate, use group convolution
        self.h1 = blocks.linear_with_channels(config_dict["num_cosmo_params"] + config_dict["num_bias_params"],
                                              config_dict["mlp_dims"][0], self.num_zbins)

        self.mlp_blocks = nn.Sequential()
        for i in range(config_dict["num_mlp_blocks"]):
            self.mlp_blocks.add_module("ResNet"+str(i+1),
                    blocks.block_parallel_resnet(config_dict["mlp_dims"][i],
                                        config_dict["mlp_dims"][i+1],
                                        config_dict["num_block_layers"],
                                        self.num_zbins,
                                        config_dict["use_skip_connection"]))
        
        #self.out_ell1 = nn.Linear(config_dict["mlp_dims"][-1], config_dict["output_kbins"])
        #self.out_ell2 = nn.Linear(config_dict["mlp_dims"][-1], config_dict["output_kbins"])
        self.h2 = blocks.linear_with_channels(config_dict["mlp_dims"][-1], output_dim, self.num_zbins)

    # NOTE: This function assumes that the bounds are the same for every redshift bin
    def normalize(self, params):
        return (params - self.bounds[0]) / (self.bounds[1] - self.bounds[0])

    def organize_params(self, params):
        """returns an expanded list of parameters with each bias parameter repeated
        num_samples times (shuffled)"""

        if self.num_zbins == 1: return params
        new_params = torch.zeros((params.shape[0], self.num_zbins, self.num_cosmo_params+(self.num_bias_params*2)))

        for i in range(self.num_zbins):
            idx = 0
            for isample1, isample2 in itertools.product(range(1), repeat=2):
                if isample1 > isample2: continue
                new_params[:, idx, :self.num_cosmo_params] = params[:,:self.num_cosmo_params]
                idx1 = self.num_cosmo_params + (isample1*self.num_bias_params)
                idx2 = self.num_cosmo_params + (isample2*self.num_bias_params)
                new_params[:, idx, self.num_cosmo_params::2] = params[:,idx1:idx1+self.num_bias_params]
                new_params[:, idx, self.num_cosmo_params+1::2] = params[:,idx2:idx2+self.num_bias_params]
                idx+=1
        return new_params

    def forward(self, X):
        X = self.organize_params(X)
        X = self.normalize(X)
        X = X.flatten(1, 2)#.permute(1, 0, 2)

        X = F.relu(self.h1(X))
        for block in self.mlp_blocks:
            X = F.relu(block(X))

        X = torch.sigmoid(self.h2(X))

        return X