import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import yaml

import spherex_emu.models.network_single_tracer as network_single_tracer
from spherex_emu.utils import load_config_file

class pk_emulator():
    """Class defining the neural network emulator."""

    def __init__(self, config_dir):

        self.config_dict = load_config_file(config_dir)

        self._init_model()

    def load_model(self):
        """Loads pre-trained network"""

    def train(self):
        """Trains the network"""

    def get_pk(self):
        print("hello!")

    # -----------------------------------------------------------
    # Helper methods: Not meant to be called by the user directly
    # -----------------------------------------------------------

    def _init_model(self):
        """Initializes the network"""
        if self.config_dict.model == "MLP_single_tracer":
            self.model = network_single_tracer(self.config_dict)
        else:
            print("ERROR: Invalid value for model")
            return -1

    def _save_model(self):
        """saves the current model state to file"""

    def _train_one_epoch(self):
        """basic training loop"""