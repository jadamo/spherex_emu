import torch
import torch.nn as nn
import math
import itertools

import spherex_emu.models.blocks as blocks

class single_transformer(nn.Module):
    """Class defining a single independent transformer network"""

    def __init__(self, config_dict, is_cross_spectra:bool):
        """Initializes an individual network, responsible for outputting a portion of the full model vector. The user is not meant to call this function directly.

        Args:
            config_dict: input dictionary with various network architecture options.
            is_cross_spectra: specifies whether the given network is responsible for outputting an auto power spectrum or cross spectrum.
                Either case has slightly different input parameter sizes.
        """
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

        # mlp blocks
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
        self.embedding_layer = nn.Linear(config_dict["mlp_dims"][-1], embedding_dim)

        # do one transformer block per z-bin for now
        self.transformer_blocks = nn.Sequential()
        for i in range(config_dict["num_transformer_blocks"]):
            self.transformer_blocks.add_module("Transformer"+str(i+1),
                    blocks.block_transformer_encoder(embedding_dim, split_dim, 0.1))
            self.transformer_blocks.add_module("Activation"+str(i+1), 
                    blocks.activation_function(embedding_dim))

        self.output_layer = nn.Linear(embedding_dim, self.output_dim)

    def forward(self, input_params):
        """Passes an input tensor through the network"""

        if not self.is_cross_spectra:
            input_params = input_params[:, :-self.num_bias_params]
        
        X = self.input_layer(input_params)
        X = self.mlp_blocks(X)
        X = self.embedding_layer(X)
        X = self.transformer_blocks(X)
        X = self.output_layer(X)

        return X

class stacked_transformer(nn.Module):
    """Class defining a stack of single_transformer objects, one for each portion of the power spectrum output"""

    def __init__(self, config_dict):
        """Initializes a group of single_transformer based on the input dictionary.
        
        This function creates nz*nps total networks, where nz is the number of redshift bins, and nps
        is the number of auto + cross power spectra per redshift bin.

        Args:
            config_dict: input dictionary with various network architecture options.  
        """
        super().__init__()

        # output dimensions
        self.num_zbins = config_dict["num_zbins"]
        self.num_spectra = config_dict["num_samples"] +  math.comb(config_dict["num_samples"], 2)
        self.num_samples = config_dict["num_samples"]
        self.num_ells = config_dict["num_ells"]
        self.num_kbins = config_dict["num_kbins"]

        self.num_cosmo_params = config_dict["num_cosmo_params"]
        self.num_bias_params = config_dict["num_bias_params"]

        # Stores networks sequentially in a list
        self.networks = nn.ModuleList()
        for z in range(self.num_zbins):
            for isample1, isample2 in itertools.product(range(self.num_samples), repeat=2):
                if isample1 > isample2: continue
                self.networks.append(single_transformer(config_dict, (isample1 != isample2)))

    def organize_parameters(self, input_params):
        """Organizes input cosmology + bias parameters into a form the rest of the network expects
        
        Args:
            input_params: tensor of input parameters with shape [batch, num_cosmo_params*(num_bias_params*num_zbins*num_tracers)]
        Returns:
            organized_params: tensor of input parameters with shape [batch, num_spectra*num_zbins, num_cosmo_params + (2*self.num_bias_params)].
                The bias parameters are split corresponding to their respective redshift / tracer bin
        """

        # parameters shape is (b, nz*nps, num_cosmo*2*num_bias)
        organized_params = torch.zeros((input_params.shape[0],
                                       self.num_spectra * self.num_zbins, 
                                       self.num_cosmo_params + (2*self.num_bias_params)),
                                       device=input_params.device)

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

                organized_params[:, iter, self.num_cosmo_params:self.num_cosmo_params+self.num_bias_params] \
                    = input_params[:, self.num_cosmo_params+idx_1::iterate]
                organized_params[:, iter, self.num_cosmo_params+self.num_bias_params:self.num_cosmo_params+2*self.num_bias_params] \
                    = input_params[:, self.num_cosmo_params+idx_2::iterate]
                iter+=1

        return organized_params

    def forward(self, input_params, net_idx = None):
        """Passes an input tensor through the network"""
        
        # feed parameters through all sub-networks
        if net_idx == None:
            X = torch.zeros((input_params.shape[0], self.num_spectra, self.num_zbins, self.num_ells*self.num_kbins), 
                             device=input_params.device)
            
            for (z, ps) in itertools.product(range(self.num_zbins), range(self.num_spectra)):
                idx = (z * self.num_spectra) + ps
                X[:, ps, z] = self.networks[idx](input_params[:,idx])
    
        # feed parameters through an individual sub-network (used in training)
        else:
            X = self.networks[net_idx](input_params[:,net_idx])

        return X
        