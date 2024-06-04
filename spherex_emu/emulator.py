import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import yaml, time

from spherex_emu.models import blocks
from spherex_emu.models.single_sample_single_redshift import MLP_single_sample_single_redshift
from spherex_emu.models.single_sample_multi_redshift import MLP_single_sample_multi_redshift
from spherex_emu.dataset import pk_galaxy_dataset
from spherex_emu.utils import load_config_file, calc_avg_loss, un_normalize

class pk_emulator():
    """Class defining the neural network emulator."""

    def __init__(self, config_file="", config_dict:dict=None):

        if config_dict is not None: self.config_dict = config_dict
        else:                       self.config_dict = load_config_file(config_file)

        # load dictionary entries into their own class variables
        for key in self.config_dict:
            setattr(self, key, self.config_dict[key])

        self._init_device()
        self._init_model()
        self.model.apply(self._init_weights)

    def load_trained_model(self, path=""):
        """loads the pre-trained layers from a file into the current model
        
        Args:
            path: The directory+filename of the trained network to load. 
            If blank, uses "save_dir" found in the object's config dictionary
        """
        # pre_trained_dict = torch.load(path+"network.params")

        # for name, param in pre_trained_dict.items():
        #     if name not in self.model.state_dict():
        #         continue
        #     self.model.state_dict()[name].copy_(param)
        if path == "": path = self.save_dir
        self.model.eval()
        self.model.load_state_dict(torch.load(path+'network.params', map_location=self.device))
        self.output_normalizations = torch.load(path+"output_normalization.dat", map_location=self.device)

    def train(self, print_progress = True):
        """Trains the network"""
        train_loader = self._load_data("training", self.training_set_fraction)
        valid_loader = self._load_data("validation")

        self.train_loss = []
        self.valid_loss = []
        best_loss = torch.inf

        # loop thru training rounds
        for round in range(len(self.learning_rate)):
            if print_progress: print("Round {:0.0f}, initial learning rate = {:0.2e}".format(
                                      round, self.learning_rate[round]))
            
            if round != 0: self.load_trained_model()
            self._set_optimizer(round)

            # loop thru epochs
            epochs_since_update = 0
            for epoch in range(self.num_epochs):

                self._train_one_epoch(train_loader)
                self.train_loss.append(calc_avg_loss(self.model, train_loader))
                self.valid_loss.append(calc_avg_loss(self.model, valid_loader))

                if self.valid_loss[-1] < best_loss:
                    best_loss = self.valid_loss[-1]
                    self._save_model()
                    epochs_since_update = 0
                else:
                    epochs_since_update += 1

                if print_progress: print("Epoch : {:d}, avg train loss: {:0.4f}\t avg validation loss: {:0.4f}\t ({:0.0f})".format(epoch, self.train_loss[-1], self.valid_loss[-1], epochs_since_update))
                if epochs_since_update > self.early_stopping_epochs:
                    print("Model has not impvored for {:0.0f} epochs. Initiating early stopping...".format(epochs_since_update))
                    break

            print("Best validation loss was {:0.4f} after {:0.0f} epochs".format(
                best_loss, epoch - epochs_since_update))

    def get_power_spectra(self, params):
        """Gets the power spectra corresponding to the given params by passing them thru the network"""

        params = self._check_params(params)

        pk = self.model.forward(params)
        pk = pk.view(self.num_zbins, self.num_samples, 2, self.output_kbins)
        pk = un_normalize(pk, self.output_normalizations)
        pk = pk.to("cpu").detach().numpy()
        return pk

    # -----------------------------------------------------------
    # Helper methods: Not meant to be called by the user directly
    # -----------------------------------------------------------

    def _init_device(self):
        """Sets emulator device based on machine configuration"""
        if self.use_gpu == False:               self.device = torch.device('cpu')
        elif torch.cuda.is_available():         self.device = torch.device('cuda:0')
        elif torch.backends.mps.is_available(): self.device = torch.device("mps")
        else:                                   self.device = torch.device('cpu')

    def _init_model(self):
        """Initializes the network"""
        if self.model_type == "MLP_single_sample_single_redshift":
            self.model = MLP_single_sample_single_redshift(self.config_dict).to(self.device)
        elif self.model_type == "MLP_single_sample_multi_redshift":
            self.model = MLP_single_sample_multi_redshift(self.config_dict).to(self.device)
        else:
            print("ERROR: Invalid value for model")
            return -1
        
        self.output_normalizations = torch.cat((torch.zeros((self.num_zbins, self.num_samples, 2, 1)),
                                                torch.ones((self.num_zbins, self.num_samples, 2, 1)))).to(self.device)
        
    def _init_weights(self, m):
        """Initializes weights using a specific scheme set in the input yaml file
        
        This function is meant to be called by the constructor only.
        Current options for initialization schemes are ["normal", "He", "xavier"]
        """
        if isinstance(m, nn.Linear):
            if self.weight_initialization == "He":
                nn.init.kaiming_uniform_(m.weight)
            elif self.weight_initialization == "normal":
                nn.init.normal_(m.weight, mean=0., std=0.1)
                nn.init.zeros_(m.bias)
            elif self.weight_initialization == "xavier":
                nn.init.xavier_normal_(m.weight)
            else: # if scheme is invalid, use normal initialization as a substitute
                nn.init.normal_(m.weight, mean=0., std=0.1)
                nn.init.zeros_(m.bias)
        elif isinstance(m, blocks.linear_with_channels):
            m.initialize_params(self.weight_initialization)

    def _set_optimizer(self, round):
        if self.optimizer_type == "Adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), 
                                              lr=self.learning_rate[round])
        else:
            print("Error! Invalid optimizer type specified!")

    def _save_model(self):
        """saves the current model state to file"""
        training_data = torch.vstack([ torch.Tensor(self.train_loss), 
                                      torch.Tensor(self.valid_loss)])
        torch.save(training_data, self.save_dir+"train_data.dat")
        
        with open(self.save_dir+'config.yaml', 'w') as outfile:
            yaml.dump(dict(self.config_dict), outfile, sort_keys=False, default_flow_style=False)

        torch.save(self.output_normalizations, self.save_dir+"output_normalization.dat")
        torch.save(self.model.state_dict(), self.save_dir+'network.params')

    def _load_data(self, key, data_frac=1.0, return_dataloader=True):

        if key in ["training", "validation", "testing"]:
            data = pk_galaxy_dataset(self.training_dir, key, data_frac)
            data.to(self.device)
            data_loader = torch.utils.data.DataLoader(data, batch_size=self.config_dict["batch_size"], shuffle=True)
            
            # set normalization based on min and max values in the training set
            if key == "training":
                self.output_normalizations = data.normalizations

            if return_dataloader: return data_loader
            else: return data

    def _check_params(self, params):
        """checks that input parameters are in the expected format and within the specified boundaries"""

        if isinstance(params, torch.Tensor): params = params.to(self.device)
        else: params = torch.from_numpy(params).to(torch.float32).to(self.device)

        # for now, assume that params should be in the shape [nz, npar, num_params]
        if self.num_zbins == 1 and self.num_samples == 1: 

            assert params.shape[0] == self.num_cosmo_params + self.num_bias_params
            assert torch.all(params >= self.model.bounds[:,0]) and \
                   torch.all(params <= self.model.bounds[:,1])
        
        else:
            assert params.shape[:] == (self.num_zbins, self.num_samples, self.num_cosmo_params + self.num_bias_params)
            # # TODO: replace this with faster code
            for z in range(self.num_zbins):
                for s in range(self.num_samples):
                    # check cosmology parameters 
                    assert torch.all(params[z,s,:] >= self.model.bounds[:,0]) and \
                           torch.all(params[z,s,:] <= self.model.bounds[:,1])
        
        return params.unsqueeze(0)

    def _train_one_epoch(self, train_loader):
        """basic training loop"""
        self.model.train()

        total_loss = 0.
        for (i, batch) in enumerate(train_loader):
            params = train_loader.dataset.get_repeat_params(batch[2], self.num_zbins, self.num_samples)
            #params = batch[0]
            target = batch[1]

            prediction = self.model.forward(params)

            loss = F.mse_loss(prediction, target, reduction="sum")
            self.optimizer.zero_grad()
            loss.backward()

            self.optimizer.step()
            total_loss += loss
        
        #self.train_loss.append(total_loss / len(train_loader))

    