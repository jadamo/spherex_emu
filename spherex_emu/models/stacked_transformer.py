import torch
import torch.nn as nn
from torch.nn import functional as F
import math, time
import itertools
from spherex_emu.utils import normalize_cosmo_params

import spherex_emu.models.blocks as blocks

class single_transformer(nn.Module):

    def __init__(self, config_dict, is_cross_spectra:bool):
        super().__init__()

        # TODO: Allow specification of activation function        
        self.num_ells = config_dict["num_ells"]
        self.num_kbins = config_dict["num_kbins"]

        self.num_bias_params = config_dict["num_bias_params"]

        # size of input depends on wether or not the network is for the crosss spectra
        self.is_cross_spectra = is_cross_spectra
        if not is_cross_spectra:
            self.input_dim = config_dict["num_cosmo_params"] + config_dict["num_bias_params"]
        else:
            self.input_dim = config_dict["num_cosmo_params"] + (2 * config_dict["num_bias_params"])
        self.output_dim = self.num_ells * self.num_kbins

        self.input_layer = nn.Linear(self.input_dim, config_dict["mlp_dims"][0])

        self.mlp_blocks = nn.Sequential()
        for i in range(config_dict["num_mlp_blocks"]):
            self.mlp_blocks.add_module("ResNet"+str(i+1),
                    blocks.block_resnet(config_dict["mlp_dims"][i],
                                        config_dict["mlp_dims"][i+1],
                                        config_dict["num_block_layers"],
                                        config_dict["use_skip_connection"]))
        
        # expand mlp section output
        split_dim = config_dict["split_dim"]
        split_size = config_dict["split_size"]
        embedding_dim = split_size*split_dim
        self.embedding_layer = nn.Linear(config_dict["mlp_dims"][-1],
                                         embedding_dim)
        #self.embedding_layer = nn.Linear(config_dict["mlp_dims"][0], embedding_dim)

        # do one transformer block per z-bin for now
        self.transformer_blocks = nn.Sequential()
        for i in range(config_dict["num_transformer_blocks"]):
            self.transformer_blocks.add_module("Transformer"+str(i+1),
                    blocks.block_transformer_encoder(embedding_dim, split_dim, 0.1))
            self.transformer_blocks.add_module("Activation"+str(i+1), 
                    blocks.activation_function(embedding_dim))

        self.output_layer = nn.Linear(embedding_dim, self.output_dim)
        #self.output_layer = nn.Linear(embedding_dim, self.output_dim)
        #self.output_activation = blocks.activation_function(self.output_dim)

    def forward(self, input_params):
        
        if not self.is_cross_spectra:
            input_params = input_params[:, :-self.num_bias_params]
        
        X = self.input_layer(input_params)
        X = self.mlp_blocks(X)
        X = self.embedding_layer(X)
        X = self.transformer_blocks(X)
        X = self.output_layer(X)

        #X = X.view(-1, self.num_zbins, self.num_spectra, self.num_ells, self.num_kbins)
        #X = torch.permute(X, (0, 1, 2, 4, 3))
        #X = X.reshape(-1, self.num_zbins, self.num_spectra * self.num_kbins * self.num_ells)

        #X = X.view(-1, self.num_zbins, self.num_spectra * self.num_ells * self.num_kbins)
        return X

class stacked_transformer(nn.Module):
    def __init__(self, config_dict):
        super().__init__()

        self.num_zbins = config_dict["num_zbins"]
        self.num_spectra = config_dict["num_samples"] +  math.comb(config_dict["num_samples"], 2)
        self.num_samples = config_dict["num_samples"]

        self.num_ells = config_dict["num_ells"]
        self.num_kbins = config_dict["num_kbins"]

        self.num_cosmo_params = config_dict["num_cosmo_params"]
        self.num_bias_params = config_dict["num_bias_params"]

        self.networks = nn.ModuleList()
        for z in range(self.num_zbins):
            for isample1, isample2 in itertools.product(range(self.num_samples), repeat=2):
                if isample1 > isample2: continue
                self.networks.append(single_transformer(config_dict, (isample1 != isample2)))

    def organize_parameters(self, input_params):
        # (b, nz*nps, num_cosmo*2*num_bias)
        organized_params = torch.zeros((input_params.shape[0],
                                       self.num_zbins * self.num_spectra, 
                                       self.num_cosmo_params + (2*self.num_bias_params)),
                                       device=input_params.device)
        #for idx in range(organized_parms.shape[1]):
        # fill cosmology parameters (the same for every bin)
        organized_params[:,:, :self.num_cosmo_params] = input_params[:, :self.num_cosmo_params].unsqueeze(1)
        # fill bias params
        # ordering is [params for tracer 1, params for tracer 2]
        iter = 0
        for z in range(self.num_zbins):
            for isample1, isample2 in itertools.product(range(self.num_samples), repeat=2):
                if isample1 > isample2: continue
                
                idx_1 = (z*self.num_samples) + isample1
                idx_2 = (z*self.num_samples) + isample2
                iterate = self.num_samples*self.num_zbins
                iterate = 4
                organized_params[:, iter, self.num_cosmo_params:self.num_cosmo_params+self.num_bias_params] \
                    = input_params[:, self.num_cosmo_params+idx_1::iterate]
                organized_params[:, iter, self.num_cosmo_params+self.num_bias_params:self.num_cosmo_params+2*self.num_bias_params] \
                    = input_params[:, self.num_cosmo_params+idx_2::iterate]
                iter+=1

        return organized_params
    
    # def parameters(self, net_idx = None):
    #     """Overloads basie parameters() function to returns the parameters of a specific sub-network"""
    #     if net_idx == None: return self.state_dict()
    #     else :              return self.networks[net_idx].parameters()

    def forward(self, input_params, net_idx = None):

        #split_params = self.organize_parameters(input_params)
        #split_params = normalize_cosmo_params(split_params)

        # feed parameters through all sub-networks
        if net_idx == None:
            X = torch.zeros((input_params.shape[0], self.num_zbins, self.num_spectra, self.num_ells*self.num_kbins), 
                            device=input_params.device)
            for z in range(self.num_zbins):
                for nps in range(self.num_spectra):
                    idx = (z * self.num_spectra) + nps
                    X[:, z, nps] = self.networks[idx](input_params[:,idx])

            X = X.reshape(-1, self.num_zbins*self.num_spectra, self.num_kbins * self.num_ells)
    
        # feed parameters through an individual sub-network (used in training)
        else:
            X = self.networks[net_idx](input_params[:,net_idx])
            X = X.reshape(-1, 1, 1*self.num_kbins * self.num_ells)

        return X
        