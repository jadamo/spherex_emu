import torch
import torch.nn as nn
from torch.nn import functional as F
import math, time

import spherex_emu.models.blocks as blocks

class transformer(nn.Module):

    def __init__(self, config_dict, device):
        super().__init__()

        # TODO: Allow specification of activation function
        self.num_zbins = config_dict["num_zbins"]
        self.num_spectra = config_dict["num_samples"] +  math.comb(config_dict["num_samples"], 2)
        self.num_ells = config_dict["num_ells"]
        self.num_kbins = config_dict["num_kbins"]

        self.input_dim = config_dict["num_cosmo_params"] + (self.num_zbins * config_dict["num_samples"] * config_dict["num_bias_params"])
        self.output_dim = self.num_spectra * self.num_ells * self.num_kbins

        # to keep each redshift bin part seperate, use group convolution
        self.input_layer = blocks.linear_with_channels(config_dict["num_cosmo_params"] + (config_dict["num_bias_params"]*self.num_zbins*config_dict["num_samples"]),
                                                       config_dict["mlp_dims"][0], 
                                                       self.num_zbins)

        self.mlp_blocks = nn.Sequential()
        for i in range(config_dict["num_mlp_blocks"]):
            self.mlp_blocks.add_module("ResNet"+str(i+1),
                    blocks.block_parallel_resnet(config_dict["mlp_dims"][i],
                                        config_dict["mlp_dims"][i+1],
                                        config_dict["num_block_layers"],
                                        self.num_zbins,
                                        config_dict["use_skip_connection"]))

        # #self.input_activation = blocks.activation_function(config_dict["mlp_dims"][0])

        # self.mlp_blocks = nn.Sequential()
        # for i in range(config_dict["num_mlp_blocks"]):
        #     self.mlp_blocks.add_module("ResMLP"+str(i+1),
        #             blocks.block_resmlp(config_dict["mlp_dims"][i],
        #                                 config_dict["mlp_dims"][i+1],
        #                                 config_dict["num_block_layers"],
        #                                 config_dict["use_skip_connection"]))
        #     self.mlp_blocks.add_module("Activation"+str(i+1), blocks.activation_function(config_dict["mlp_dims"][i+1]))

        # expand mlp section output
        split_dim = config_dict["split_dim"]
        split_size = config_dict["split_size"]
        embedding_dim = split_size*split_dim
        self.embedding_layer = blocks.linear_with_channels(config_dict["mlp_dims"][-1],
                                                           embedding_dim, 
                                                           self.num_zbins)
        #self.embedding_layer = nn.Linear(config_dict["mlp_dims"][0], embedding_dim)

        # do one transformer block per z-bin for now
        self.transformer_blocks = []
        for z in range(self.num_zbins):
            self.transformer_blocks.append(nn.Sequential())
            for i in range(config_dict["num_transformer_blocks"]):
                self.transformer_blocks[z].add_module("Transformer"+str(i+1),
                        blocks.block_transformer_encoder(embedding_dim, split_dim, 0.1))
                self.transformer_blocks[z].add_module("Activation"+str(i+1), blocks.activation_function(embedding_dim))
                self.transformer_blocks[z].to(device)

        self.output_layer = blocks.linear_with_channels(embedding_dim, self.output_dim, self.num_zbins)
        #self.output_layer = nn.Linear(embedding_dim, self.output_dim)
        #self.output_activation = blocks.activation_function(self.output_dim)

    def forward(self, input_params):

        input_params = input_params.unsqueeze(1).repeat(1, self.num_zbins, 1)
        X = self.input_layer(input_params)
        X = self.mlp_blocks(X)
        X = self.embedding_layer(X)

        # TODO: Come up with faster way to loop thru transformer blocks
        # Need to assign output to new variable to allow back-propagation
        X2 = torch.zeros_like(X)
        for z in range(self.num_zbins):
            X2[:,z,:] = self.transformer_blocks[z](X[:,z,:])
        X = self.output_layer(X2)

        X = X.view(-1, self.num_zbins, self.num_spectra, self.num_ells, self.num_kbins)
        X = torch.permute(X, (0, 1, 2, 4, 3))
        X = X.reshape(-1, self.num_zbins, self.num_spectra * self.num_kbins * self.num_ells)

        #X = X.view(-1, self.num_zbins, self.num_spectra * self.num_ells * self.num_kbins)
        return X
