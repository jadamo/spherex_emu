import torch
import torch.nn as nn
from torch.nn import functional as F
import itertools, math

import spherex_emu.models.blocks as blocks
from spherex_emu.utils import get_parameter_ranges, load_config_file, un_normalize

class mlp(nn.Module):

    def __init__(self, config_dict):
        super().__init__()

        output_dim = config_dict["num_kbins"] * config_dict["num_ells"]
        self.num_zbins = config_dict["num_zbins"]
        self.num_spectra = config_dict["num_samples"] +  math.comb(config_dict["num_samples"], 2)
        self.num_ells = config_dict["num_ells"]
        self.num_kbins = config_dict["num_kbins"]
        self.num_cosmo_params = config_dict["num_cosmo_params"]
        self.num_bias_params  = config_dict["num_bias_params"]
        self.output_normalizations = None

        cosmo_file = load_config_file(config_dict["input_dir"] + config_dict["cosmo_dir"])
        __, bounds = get_parameter_ranges(cosmo_file)
        self.register_buffer("bounds", self.organize_params(torch.Tensor(bounds.T)))

        # to keep each redshift bin part seperate, use group convolution
        self.h1 = blocks.linear_with_channels(config_dict["num_cosmo_params"] + (config_dict["num_bias_params"]*2),
                                              config_dict["mlp_dims"][0], self.num_zbins*self.num_spectra)

        self.mlp_blocks = nn.Sequential()
        for i in range(config_dict["num_mlp_blocks"]):
            self.mlp_blocks.add_module("ResNet"+str(i+1),
                    blocks.block_parallel_resnet(config_dict["mlp_dims"][i],
                                        config_dict["mlp_dims"][i+1],
                                        config_dict["num_block_layers"],
                                        self.num_zbins*self.num_spectra,
                                        config_dict["use_skip_connection"]))
        
        #self.out_ell1 = nn.Linear(config_dict["mlp_dims"][-1], config_dict["num_kbins"])
        #self.out_ell2 = nn.Linear(config_dict["mlp_dims"][-1], config_dict["num_kbins"])
        self.h2 = blocks.linear_with_channels(config_dict["mlp_dims"][-1], output_dim, self.num_zbins*self.num_spectra)

    def set_normalizations(self, output_normalizations):
        self.output_normalizations = output_normalizations

    def organize_params(self, params):
        """returns an expanded list of parameters with each bias parameter repeated
        num_samples times (shuffled)"""

        if self.num_zbins*self.num_spectra == 1: return params
        new_params = torch.zeros((params.shape[0], self.num_zbins*self.num_spectra, self.num_cosmo_params+(self.num_bias_params*2)))

        for i in range(self.num_zbins*self.num_spectra):
            idx = 0
            for isample1, isample2 in itertools.product(range(self.num_spectra), repeat=2):
                if isample1 > isample2: continue
                # fill in cosmology parameters
                new_params[:, idx, :self.num_cosmo_params] = params[:,:self.num_cosmo_params]
                # fill in relavent bias parameters
                idx1 = self.num_cosmo_params + (isample1*self.num_bias_params)
                idx2 = self.num_cosmo_params + (isample2*self.num_bias_params)
                new_params[:, idx, self.num_cosmo_params::2] = params[:,idx1:idx1+self.num_bias_params]
                new_params[:, idx, self.num_cosmo_params+1::2] = params[:,idx2:idx2+self.num_bias_params]
                idx+=1
        return new_params

    def forward(self, X):
        X = self.organize_params(X)
        #X = X.flatten(1, 2)#.permute(1, 0, 2)

        X = F.relu(self.h1(X))
        X = F.relu(self.mlp_blocks(X))
        X = torch.sigmoid(self.h2(X))

        X = X.view(-1, self.num_zbins, self.num_spectra, self.num_ells, self.num_kbins)
        X = un_normalize(X, self.output_normalizations)

        return X
