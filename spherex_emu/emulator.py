import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import yaml, time

from spherex_emu.models.single_tracer import MLP_single_tracer
from spherex_emu.dataset import pk_galaxy_dataset
from spherex_emu.utils import load_config_file, calc_avg_loss

class pk_emulator():
    """Class defining the neural network emulator."""

    def __init__(self, config_file):

        self.config_dict = load_config_file(config_file)

        self._init_model()
        self.model.apply(self._init_weights)

    def load_model(self):
        """Loads pre-trained network"""

    def train(self):
        """Trains the network"""
        train_loader = self._load_data("training")
        valid_loader = self._load_data("validation")

        self.train_loss = []
        self.valid_loss = []

        self._set_optimizer()

        best_loss = torch.inf
        epochs_since_update = 0
        t1 = time.time()
        for epoch in range(self.config_dict.num_epochs):

            self._train_one_epoch(train_loader)
            self.valid_loss.append(calc_avg_loss(self.model, valid_loader))

            if self.valid_loss[-1] < best_loss:
                best_loss = self.valid_loss[-1]
                self._save_model()
                epochs_since_update = 0
            else:
                epochs_since_update += 1

            print("Epoch : {:d}, avg train loss: {:0.3f}\t avg validation loss: {:0.3f}\t ({:0.0f})".format(epoch, self.train_loss[-1], self.valid_loss[-1], epochs_since_update))
            if epochs_since_update > self.config_dict.early_stopping_epochs:
                print("Model has not impvored for {:0.0f} epochs. Initiating early stopping...".format(epochs_since_update))
                break

        print("Best validation loss was {:0.3f} after {:0.0f} epochs".format(
               best_loss, epoch - epochs_since_update))


    def get_pk(self, params):
        print("hello!")

    # -----------------------------------------------------------
    # Helper methods: Not meant to be called by the user directly
    # -----------------------------------------------------------

    def _init_model(self):
        """Initializes the network"""
        if self.config_dict.model == "MLP_single_tracer":
            self.model = MLP_single_tracer(self.config_dict)
        else:
            print("ERROR: Invalid value for model")
            return -1
        
        self.training_data = []

    def _init_weights(self, m):
        """Initializes weights using a specific scheme set in the input yaml file
        
        This function is meant to be called by the constructor only.
        Current options for initialization schemes are ["normal", "He", "xavier"]
        """
        if isinstance(m, nn.Linear):
            if self.config_dict.weight_initialization == "He":
                nn.init.kaiming_uniform_(m.weight)
            elif self.config_dict.weight_initialization == "normal":
                nn.init.normal_(m.weight, mean=0., std=0.1)
                nn.init.zeros_(m.bias)
            elif self.config_dict.weight_initialization == "xavier":
                nn.init.xavier_normal_(m.weight)
            else: # if scheme is invalid, use normal initialization as a substitute
                nn.init.normal_(m.weight, mean=0., std=0.1)
                nn.init.zeros_(m.bias)

    def _set_optimizer(self):
        if self.config_dict.optimizer == "Adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), 
                                              lr=self.config_dict.learning_rate[0])
        else:
            print("Error! Invalid optimizer type specified!")

    def _save_model(self):
        """saves the current model state to file"""
        training_data = torch.vstack([ torch.Tensor(self.train_loss), 
                                      torch.Tensor(self.valid_loss)])
        torch.save(training_data, self.config_dict.save_dir+"train_data.dat")

        with open(self.config_dict.save_dir+'config.yaml', 'w') as outfile:
            yaml.dump(dict(self.config_dict), outfile, default_flow_style=False)

        torch.save(self.model.state_dict(), self.config_dict.save_dir+'network.params')

    def _load_data(self, key):

        if key in ["training", "validation", "testing"]:
            data = pk_galaxy_dataset(self.config_dict.training_dir, key, 1.)
            data_loader = torch.utils.data.DataLoader(data, batch_size=self.config_dict.batch_size, shuffle=True)
            return data_loader

    def _train_one_epoch(self, train_loader):
        """basic training loop"""
        self.model.train()

        total_loss = 0.
        for (i, batch) in enumerate(train_loader):
            params = batch[0]
            target = batch[1]

            prediction = self.model.forward(params)

            loss = F.mse_loss(prediction, target, reduction="sum")
            self.optimizer.zero_grad()
            loss.backward()

            self.optimizer.step()
            total_loss += loss
        
        self.train_loss.append(total_loss / len(train_loader))

    