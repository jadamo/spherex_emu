import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import yaml, math, os

from spherex_emu.models import blocks
from spherex_emu.models.mlp import mlp
from spherex_emu.models.transformer import transformer
from spherex_emu.dataset import pk_galaxy_dataset
from spherex_emu.utils import load_config_file, calc_avg_loss, get_parameter_ranges,\
                              normalize_cosmo_params, un_normalize_power_spectrum, \
                              delta_chi_squared, mse_loss, hyperbolic_loss, hyperbolic_chi2_loss
from spherex_emu.filepaths import base_dir, data_dir

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
        self._init_loss()
        self._init_fiducial_power_spectrum()
        self._init_inverse_covariance()
        self._diagonalize_covariance()
        self._init_normalizations() # <- This function might not be necesary anymore
        self.model.apply(self._init_weights)

    def load_trained_model(self, path=""):
        """loads the pre-trained layers from file into the current model, as well as all relavent information needed for normalization
        
        Args:
            path: The directory+filename of the trained network to load. 
            If blank, uses "save_dir" found in the object's config dictionary
        """
        if path == "": path = base_dir+self.save_dir
        self.model.eval()
        self.model.load_state_dict(torch.load(path+'network.params', map_location=self.device))
        
        self.ps_fid = torch.load(path+"ps_fid.dat", map_location=self.device)
        self.invcov = torch.load(path+"invcov.dat", map_location=self.device)
        self.eigvals = torch.load(path+"eigenvals.dat", map_location=self.device)
        self.Q = torch.load(path+"eigenvectors.dat", map_location=self.device)
        for z in range(self.num_zbins):
            self.Q_inv[z] = torch.linalg.inv(self.Q[z])
        #self.output_normalizations = torch.load(path+"output_normalization.dat", map_location=self.device)

    def load_data(self, key, data_frac=1.0, return_dataloader=True, data_dir=""):

        if data_dir != "": dir = data_dir
        else :             dir = base_dir+self.training_dir

        if key in ["training", "validation", "testing"]:
            data = pk_galaxy_dataset(dir, key, data_frac)
            data.to(self.device)
            data_loader = torch.utils.data.DataLoader(data, batch_size=self.config_dict["batch_size"], shuffle=True)
            
            # set normalization based on min and max values in the training set
            # if key == "training":
            #     self.output_normalizations = data.output_normalizations.to(self.device)
                #self.model.set_normalizations(self.output_normalizations)

            if return_dataloader: return data_loader
            else: return data

    def train(self, print_progress = True):
        """Trains the network"""
        train_loader = self.load_data("training", self.training_set_fraction)
        valid_loader = self.load_data("validation")

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
                self.train_loss.append(calc_avg_loss(self.model, train_loader, self.input_normalizations, self.ps_fid, self.invcov, self.loss_function))
                self.valid_loss.append(calc_avg_loss(self.model, valid_loader, self.input_normalizations, self.ps_fid, self.invcov, self.loss_function))

                if self.valid_loss[-1] < best_loss:
                    best_loss = self.valid_loss[-1]
                    self._save_model()
                    epochs_since_update = 0
                else:
                    epochs_since_update += 1

                if print_progress: print("Epoch : {:d}, avg train loss: {:0.4e}\t avg validation loss: {:0.4e}\t ({:0.0f})".format(epoch, self.train_loss[-1], self.valid_loss[-1], epochs_since_update))
                if epochs_since_update > self.early_stopping_epochs:
                    print("Model has not impvored for {:0.0f} epochs. Initiating early stopping...".format(epochs_since_update))
                    break

            print("Best validation loss was {:0.4e} after {:0.0f} epochs".format(
                best_loss, epoch - epochs_since_update))

    def get_power_spectra(self, params):
        """Gets the power spectra corresponding to the given params by passing them thru the network"""

        params = self._check_params(params)
        norm_params = normalize_cosmo_params(params, self.input_normalizations)
        pk = self.model.forward(norm_params)
        pk = un_normalize_power_spectrum(pk, self.ps_fid, self.invcov)
        pk = pk.view(self.num_zbins, self.num_spectra, self.num_kbins, self.num_ells)
        pk = torch.permute(pk, (0, 1, 3, 2))

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
        self.num_spectra = self.num_samples +  math.comb(self.num_samples, 2)
        if self.model_type == "mlp":
            self.model = mlp(self.config_dict).to(self.device)
        elif self.model_type == "transformer":
            self.model = transformer(self.config_dict).to(self.device)
        else:
            print("ERROR: Invalid value for model type")
            raise KeyError
                
    def _init_normalizations(self):
        """Initializes both input and output normalization factors"""
        try:
            cosmo_dict = load_config_file(base_dir+self.cosmo_dir)
            __, bounds = get_parameter_ranges(cosmo_dict)
            self.input_normalizations = torch.Tensor(bounds.T).to(self.device)
        except IOError:
            self.input_normalizations = torch.vstack((torch.zeros((self.num_cosmo_params + (self.num_samples*self.num_zbins*self.num_bias_params))),
                                                      torch.ones((self.num_cosmo_params + (self.num_samples*self.num_zbins*self.num_bias_params))))).to(self.device)

        # NOTE: Actively changing how I'm normalizing the output!
        self.output_normalizations = torch.cat((torch.zeros((self.num_zbins, self.num_spectra, 2, 1)),
                                                torch.ones((self.num_zbins, self.num_spectra, 2, 1)))).to(self.device)
        #self.model.set_normalizations(self.output_normalizations)

    def _init_fiducial_power_spectrum(self):
        """Loads the fiducial power spectrum for use in normalization"""
        ps_file = base_dir+self.training_dir+"ps_fid.npy"
        if os.path.exists(ps_file):
            self.ps_fid = torch.from_numpy(np.load(ps_file)).to(self.device).to(torch.float32)
            if self.ps_fid.shape[3] == self.num_kbins:
                self.ps_fid = torch.permute(self.ps_fid, (0, 1, 3, 2))
            self.ps_fid = self.ps_fid.reshape(self.num_zbins, self.num_spectra * self.num_kbins * self.num_ells)
        else:
            print("WARNING: Could not load fiducial power spectrum!")
            self.ps_fid = torch.zeros((self.num_zbins, self.num_spectra * self.num_ells * self.num_kbins)).to(self.device)

    def _init_inverse_covariance(self):
        """Loads the inverse data covariance matrix for use in certain loss functions and normalizations"""
        #cov_file = data_dir+"cov_"+str(self.num_samples)+"_sample_"+str(self.num_zbins)+"_redshift/invcov_reshape.npy"
        cov_file = base_dir+self.training_dir
        if os.path.exists(cov_file+"invcov.npy"):
            self.invcov = torch.from_numpy(np.load(cov_file+"invcov.npy")).to(torch.float32).to(self.device)
        elif os.path.exists(cov_file+"invcov.dat"):
            self.invcov = torch.load(cov_file+"invcov.dat").to(self.device).to(torch.float32)
        else:
            print("WARNING: Could not load inverse covariance matrix")
            self.invcov = torch.eye(self.num_ells*self.num_spectra*self.num_kbins).unsqueeze(0)
            self.invcov = self.invcov.repeat(self.num_zbins, 1, 1).to(self.device)

    def _diagonalize_covariance(self):
        """performs an eigenvalue decomposition of the inverse covariance matrix"""
        self.Q, = torch.zeros_like(self.invcov)
        self.Q_inv = torch.zeros_like(self.invcov)
        self.eigvals = torch.zeros(self.invcov.shape[1])
        for z in range(self.num_zbins):
            q, eig = torch.linalg.eigh(self.invcov[z])
            self.Q[z] = q.real
            self.Q_inv[z] = torch.linalg.inv(q).real
            self.eigvals[z] = eig.real

        assert torch.all(self.eigvals > 0), "ERROR! covariance matrix has negative eigenvalues? Is it positive definite?"

    def _init_loss(self):
        """Defines the loss function to use"""
        if self.loss_type == "chi2":
            self.loss_function = delta_chi_squared
        elif self.loss_type == "mse":
            self.loss_function = mse_loss
        elif self.loss_type == "hyperbolic":
            self.loss_function = hyperbolic_loss
        elif self.loss_type == "hyperbolic_chi2":
            self.loss_function = hyperbolic_chi2_loss
        else:
            print("ERROR: Invalid loss function type")
            raise KeyError

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
        """saves the current model state and normalization information to file"""
        training_data = torch.vstack([ torch.Tensor(self.train_loss), 
                                      torch.Tensor(self.valid_loss)])
        torch.save(training_data, base_dir+self.save_dir+"train_data.dat")
        
        with open(base_dir+self.save_dir+'config.yaml', 'w') as outfile:
            yaml.dump(dict(self.config_dict), outfile, sort_keys=False, default_flow_style=False)

        # data needed for normalization
        torch.save(self.ps_fid, base_dir+self.save_dir+"ps_fid.dat")
        torch.save(self.invcov, base_dir+self.save_dir+"invcov.dat")
        torch.save(self.eigvals, base_dir+self.save_dir+"eigenvals.dat")
        torch.save(self.Q, base_dir+self.save_dir+"eigenvectors.dat")
        #torch.save(self.output_normalizations, base_dir+self.save_dir+"output_normalization.dat")
        torch.save(self.model.state_dict(), base_dir+self.save_dir+'network.params')

    def _check_params(self, params):
        """checks that input parameters are in the expected format and within the specified boundaries"""

        if isinstance(params, torch.Tensor): params = params.to(self.device)
        else: params = torch.from_numpy(params).to(torch.float32).to(self.device)

        # for now, assume that params should be 1D and in the form
        # [cosmo_params, bias_params for each sample / zbin grouped together]
        assert params.shape[0] == self.num_cosmo_params + (self.num_bias_params * self.num_zbins * self.num_samples)
        
        return params.unsqueeze(0)

    def _train_one_epoch(self, train_loader):
        """basic training loop"""
        self.model.train()

        total_loss = 0.
        for (i, batch) in enumerate(train_loader):
            #params = train_loader.dataset.get_repeat_params(batch[2], self.num_zbins, self.num_samples)
            params = normalize_cosmo_params(batch[0], self.input_normalizations)
            target = batch[1]

            prediction = self.model.forward(params)
            prediction = un_normalize_power_spectrum(prediction, self.ps_fid, self.invcov)

            loss = self.loss_function(prediction, target, self.invcov)
            self.optimizer.zero_grad()
            loss.backward()

            self.optimizer.step()
            total_loss += loss
