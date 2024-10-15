import torch
import torch.nn as nn
from torch.nn import functional as F
import itertools, math

import spherex_emu.models.blocks as blocks
from spherex_emu.utils import get_parameter_ranges, load_config_file, un_normalize
from spherex_emu.filepaths import base_dir

class MLP_multi_sample_multi_redshift(nn.Module):

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

        cosmo_file = load_config_file(base_dir + config_dict["cosmo_dir"])
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
        for block in self.mlp_blocks:
            X = F.relu(block(X))
        X = torch.sigmoid(self.h2(X))

        X = X.view(-1, self.num_zbins, self.num_spectra, self.num_ells, self.num_kbins)
        X = un_normalize(X, self.output_normalizations)

        return X
    

class Transformer(nn.Module):

    def __init__(self, config_dict):
        super().__init__()

        # TODO: Allow specification of activation function
        self.num_zbins = config_dict["num_zbins"]
        self.num_spectra = config_dict["num_samples"] +  math.comb(config_dict["num_samples"], 2)
        self.num_ells = config_dict["num_ells"]
        self.num_kbins = config_dict["num_kbins"]

        self.input_dim = config_dict["num_cosmo_params"] + (self.num_zbins * config_dict["num_samples"] * config_dict["num_bias_params"])
        self.output_dim = self.num_zbins * self.num_spectra * self.num_ells * self.num_kbins

        # cosmo_file = load_config_file(base_dir + config_dict["cosmo_dir"])
        # __, bounds = get_parameter_ranges(cosmo_file)
        # self.register_buffer("bounds", torch.Tensor(bounds.T))

        self.input_layer = nn.Linear(self.input_dim, config_dict["mlp_dims"][0])
        self.input_activation = blocks.activation_function(config_dict["mlp_dims"][0])

        self.mlp_blocks = nn.Sequential()
        for i in range(config_dict["num_mlp_blocks"]):
            self.mlp_blocks.add_module("ResMLP"+str(i+1),
                    blocks.block_resmlp(config_dict["mlp_dims"][i],
                                        config_dict["mlp_dims"][i+1],
                                        config_dict["num_block_layers"],
                                        config_dict["use_skip_connection"]))
            self.mlp_blocks.add_module("Activation"+str(i+1), blocks.activation_function(config_dict["mlp_dims"][i+1]))

        split_dim = config_dict["split_dim"]
        embedding_dim = self.num_ells*self.num_kbins*split_dim
        self.embedding_layer = nn.Linear(config_dict["mlp_dims"][0], embedding_dim)

        self.transformer_blocks = nn.Sequential()
        for i in range(config_dict["num_transformer_blocks"]):
            self.transformer_blocks.add_module("Transformer"+str(i+1),
                    blocks.block_transformer_encoder(embedding_dim, split_dim, 0.1))
            self.transformer_blocks.add_module("Activation"+str(i+1), blocks.activation_function(embedding_dim))

        self.output_layer = nn.Linear(embedding_dim, self.output_dim)
        self.output_activation = blocks.activation_function(self.output_dim)

    def forward(self, X):

        X = self.input_activation(self.input_layer(X))
        for block in self.mlp_blocks:
            X = block(X)

        X = self.embedding_layer(X)
        for block in self.transformer_blocks:
            X = block(X)
        X = self.output_activation(self.output_layer(X))

        X = X.view(-1, self.num_zbins, self.num_spectra * self.num_ells * self.num_kbins)
        return X
