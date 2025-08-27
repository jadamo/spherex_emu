import torch.nn as nn

import mentat_lss.models.blocks as blocks

class single_transformer(nn.Module):
    """Class defining a single independent transformer network"""

    def __init__(self, config_dict):
        """Initializes an individual network, responsible for outputting a portion of the full model vector. The user is not meant to call this function directly.

        Args:
            config_dict: input dictionary with various network architecture options.
            is_cross_spectra: specifies whether the given network is responsible for outputting an auto power spectrum or cross spectrum.
                Either case has slightly different input parameter sizes.
        """
        super().__init__()

        self.num_kbins = config_dict["ps_nw_emulator"]["num_kbins"]

        self.input_dim = config_dict["num_cosmo_params"]
        self.output_dim = self.num_kbins

        # mlp blocks
        self.input_layer = nn.Linear(self.input_dim, self.output_dim)
        self.mlp_blocks = nn.Sequential()
        for i in range(config_dict["ps_nw_emulator"]["num_mlp_blocks"]):
            self.mlp_blocks.add_module("ResNet"+str(i+1),
                    blocks.block_resnet(self.output_dim,
                                        self.output_dim,
                                        config_dict["ps_nw_emulator"]["num_block_layers"],
                                        config_dict["ps_nw_emulator"]["use_skip_connection"]))
        
        # expand mlp section output
        split_dim = config_dict["ps_nw_emulator"]["split_dim"]
        split_size = config_dict["ps_nw_emulator"]["split_size"]
        embedding_dim = split_size*split_dim
        self.embedding_layer = nn.Linear(self.output_dim, embedding_dim)

        # do one transformer block per z-bin for now
        self.transformer_blocks = nn.Sequential()
        for i in range(config_dict["ps_nw_emulator"]["num_transformer_blocks"]):
            self.transformer_blocks.add_module("Transformer"+str(i+1),
                    blocks.block_transformer_encoder(embedding_dim, split_dim, 0.1))
            self.transformer_blocks.add_module("Activation"+str(i+1), 
                    blocks.activation_function(embedding_dim))

        self.output_layer = nn.Linear(embedding_dim, self.output_dim)
        self.output_activation = nn.Sigmoid()

    def forward(self, input_params):
        """Passes an input tensor through the network"""
        
        input_params = input_params[:, :self.input_dim]

        X = self.input_layer(input_params)
        X = self.mlp_blocks(X)
        X = self.embedding_layer(X)
        X = self.transformer_blocks(X)
        X = self.output_activation(self.output_layer(X)) * 5 - 1.
     
        return X
        
