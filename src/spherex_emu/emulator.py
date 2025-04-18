import torch
import torch.nn as nn
import numpy as np
import yaml, math, os, time
import itertools

from spherex_emu.models import blocks
from spherex_emu.models.stacked_mlp import stacked_mlp
from spherex_emu.models.stacked_transformer import stacked_transformer
from spherex_emu.dataset import pk_galaxy_dataset
from spherex_emu.utils import load_config_file, calc_avg_loss, get_parameter_ranges,\
                              normalize_cosmo_params, un_normalize_power_spectrum, \
                              delta_chi_squared, mse_loss, hyperbolic_loss, hyperbolic_chi2_loss, \
                              get_invcov_blocks, get_full_invcov

class pk_emulator():
    """Class defining the neural network emulator."""


    def __init__(self, net_dir:str, mode:str="train"):
        """Emulator constructor, initializes the network structure and all supporting data

        Args:
            net_dir: string specifying either the directory or full filepath of the trained emulator to load from.
            if a directory, assumes the config file is called "config.yaml"
            mode: whether the emulator should initialize for training, or to load from a previous training run. One 
            of either ["train", "eval"]. Detailt "train"

        Raises:
            KeyError: if mode is not correctly specified
            IOError: if no input yaml file was found
        """
        if net_dir.endswith(".yaml"): self.config_dict = load_config_file(net_dir)
        else:                         self.config_dict = load_config_file(net_dir+"config.yaml")

        # load dictionary entries into their own class variables
        for key in self.config_dict:
            setattr(self, key, self.config_dict[key])

        self._init_device()
        self._init_model()
        self._init_loss()

        if mode == "train":
            self._init_fiducial_power_spectrum()
            self._init_inverse_covariance()
            self._diagonalize_covariance()
            self._init_input_normalizations()
            self.model.apply(self._init_weights)

        elif mode == "eval":
            self.load_trained_model(net_dir)

        else:
            print("ERROR! Invalid mode specified! Must be one of ['train', 'eval']")
            raise KeyError


    def load_trained_model(self, path):
        """loads the pre-trained network from file into the current model, as well as all relavent information needed for normalization.
        This function is called by the constructor, but can also be called directly by the user if desired.
        
        Args:
            path: The directory+filename of the trained network to load. 
        """

        print("loading emulator from " + path)
        self.model.eval()
        self.model.load_state_dict(torch.load(path+'network.params', map_location=self.device))
        
        self.input_normalizations = torch.load(path+"input_normalizations.pt", map_location=self.device)

        output_norm_data = torch.load(path+"output_normalizations.pt", map_location=self.device)
        self.ps_fid        = output_norm_data[0]
        self.invcov_full   = output_norm_data[1]
        self.invcov_blocks = output_norm_data[2]
        self.sqrt_eigvals  = output_norm_data[3]
        self.Q             = output_norm_data[4]
        self.Q_inv = torch.zeros_like(self.Q, device="cpu")
        for (ps, z) in itertools.product(range(self.num_spectra), range(self.num_zbins)):
            self.Q_inv[ps, z] = torch.linalg.inv(self.Q[ps, z].to("cpu").to(torch.float64)).to(torch.float32)
        self.Q_inv = self.Q_inv.to(self.device)


    def load_data(self, key: str, data_frac = 1.0, return_dataloader=True, data_dir=""):
        """loads and returns the training / validation / test dataset into memory
        
        Args:
            key: one of ["training", "validation", "testing"] that specifies what type of data-set to laod
            data_frac: fraction of the total data-set to load in. Default 1
            return_dataloader: Determines what object type to return the data as. Default True
                If true: returns data as a pytorch.utils.data.DataLoader object.
                If false: returns data as a pk_galaxy_dataset object.
            data_dir: location of the data-set on disk. Default ""

        Returns:
            data: The desired data-set in either a pk_galaxy_dataset or DataLoader object.
        """

        if data_dir != "": dir = data_dir
        else :             dir = self.input_dir+self.training_dir

        if key in ["training", "validation", "testing"]:
            data = pk_galaxy_dataset(dir, key, data_frac)
            data.to(self.device)
            data.normalize_data(self.ps_fid, self.sqrt_eigvals, self.Q)
            data_loader = torch.utils.data.DataLoader(data, batch_size=self.config_dict["batch_size"], shuffle=True)
        
            if return_dataloader: return data_loader
            else: return data


    def train(self):
        """Trains the network"""

        # load training / validation datasets
        train_loader = self.load_data("training", self.training_set_fraction)
        valid_loader = self.load_data("validation")

        # store training data as nested lists with dims [nps, nz]
        self.train_loss     = [[[] for i in range(self.num_zbins)] for j in range(self.num_spectra)]
        self.valid_loss     = [[[] for i in range(self.num_zbins)] for j in range(self.num_spectra)]
        best_loss           = [[torch.inf for i in range(self.num_zbins)] for j in range(self.num_spectra)]
        epochs_since_update = [[0 for i in range(self.num_zbins)] for j in range(self.num_spectra)]
        self.train_time = 0.
        if self.print_progress: print("Initial learning rate = {:0.2e}".format(self.learning_rate))
        
        self._set_optimizer()
        self.model.train()
        start_time = time.time()
        # loop thru epochs
        for epoch in range(self.num_epochs):
            # loop thru individual networks
            for (ps, z) in itertools.product(range(self.num_spectra), range(self.num_zbins)):
                if epochs_since_update[ps][z] > self.early_stopping_epochs:
                    continue

                training_loss = self._train_one_epoch(train_loader, [ps, z])
                if self.recalculate_train_loss:
                    self.train_loss[ps][z].append(calc_avg_loss(self.model, train_loader, self.input_normalizations, 
                                                                self.ps_fid, self.invcov_full, self.sqrt_eigvals, 
                                                                self.Q, self.Q_inv, self.loss_function, [ps, z]))
                else:
                    self.train_loss[ps][z].append(training_loss)
                self.valid_loss[ps][z].append(calc_avg_loss(self.model, valid_loader, self.input_normalizations, 
                                                            self.ps_fid, self.invcov_full, self.sqrt_eigvals, 
                                                            self.Q, self.Q_inv, self.loss_function, [ps, z]))
                
                self.scheduler[ps][z].step(self.valid_loss[ps][z][-1])
                self.train_time = time.time() - start_time

                if self.valid_loss[ps][z][-1] < best_loss[ps][z]:
                    best_loss[ps][z] = self.valid_loss[ps][z][-1]
                    epochs_since_update[ps][z] = 0
                    self._save_model()
                else:
                    epochs_since_update[ps][z] += 1

                if self.print_progress: print("Net idx : [{:d}, {:d}], Epoch : {:d}, avg train loss: {:0.4e}\t avg validation loss: {:0.4e}\t ({:0.0f})".format(
                    ps, z, epoch, self.train_loss[ps][z][-1], self.valid_loss[ps][z][-1], epochs_since_update[ps][z]))
                if epochs_since_update[ps][z] > self.early_stopping_epochs:
                    print("Model [{:d}, {:d}] has not impvored for {:0.0f} epochs. Initiating early stopping...".format(ps, z, epochs_since_update[ps][z]))


    def get_power_spectra(self, params, raw_output:bool = False):
        """Gets the power spectra corresponding to the given input params by passing them though the emulator
        
        Args:
            params: 1D numpy array or torch Tensor containing a list of cosmology + galaxy bias parameters.
            raw_output: bool specifying whether or not to return the raw network output without undoing normalization. Default False
        """

        # TODO: Add k-bin check (it should match the covariance matrix)
        self.model.eval()
        with torch.no_grad():
            params = self._check_params(params)
            pk = self.model.forward(params)[0] # <- shape [nps, nz, nk*nl]
            
            if raw_output:
                return pk
            else:
                pk = un_normalize_power_spectrum(torch.flatten(pk, start_dim=2), self.ps_fid, self.sqrt_eigvals, self.Q, self.Q_inv)
                pk = pk.view(self.num_spectra, self.num_zbins, self.num_kbins, self.num_ells)
                pk = pk.to("cpu").detach().numpy()
                return pk


    def get_required_parameters(self):
        """Returns a dictionary of input parameters needed by the emulator. Used within Cosmo_Inference"""

        # TODO: read in these requirnments from a file
        cosmo_params = ["As", "ns", "fnl"]
        bias_params = ["galaxy_bias_10", "galaxy_bias_20", "galaxy_bias_G2"]
        counterterm_params = ["counterterm_0", "counterterm_2", "counterterm_fog"]
        required_params = {"cosmo_params":cosmo_params, "galaxy_bias_params":bias_params, "counterterm_params": counterterm_params}
        return required_params

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
        self.num_spectra = self.num_tracers + math.comb(self.num_tracers, 2)
        if self.model_type == "stacked_mlp":
            self.model = stacked_mlp(self.config_dict).to(self.device)
        elif self.model_type == "stacked_transformer":
            self.model = stacked_transformer(self.config_dict).to(self.device)
        else:
            print("ERROR: Invalid value for model_type", self.model_type)
            raise KeyError
                

    def _init_input_normalizations(self):
        """Initializes input parameter normalization factors
        
        Normalizations are in the shape (low / high bound, net_idx, parameter)
        """

        try:
            cosmo_dict = load_config_file(self.input_dir+self.cosmo_dir)
            __, param_bounds = get_parameter_ranges(cosmo_dict)
            input_normalizations = torch.Tensor(param_bounds.T).to(self.device)
        except IOError:
            input_normalizations = torch.vstack((torch.zeros((self.num_cosmo_params + (self.num_tracers*self.num_zbins*self.num_nuisance_params))),
                                                 torch.ones((self.num_cosmo_params + (self.num_tracers*self.num_zbins*self.num_nuisance_params))))).to(self.device)
        
        lower_bounds = self.model.organize_parameters(input_normalizations[0].unsqueeze(0))
        upper_bounds = self.model.organize_parameters(input_normalizations[1].unsqueeze(0))
        self.input_normalizations = torch.vstack([lower_bounds, upper_bounds])


    def _init_fiducial_power_spectrum(self):
        """Loads the fiducial power spectrum for use in normalization"""

        ps_file = self.input_dir+self.training_dir+"ps_fid.npy"

        if os.path.exists(ps_file):
            self.ps_fid = torch.from_numpy(np.load(ps_file)).to(torch.float32).to(self.device)[0]

            # permute input power spectrum if it's a different shape than expected
            if self.ps_fid.shape[3] == self.num_kbins:
                self.ps_fid = torch.permute(self.ps_fid, (0, 1, 3, 2))
            if self.ps_fid.shape[0] == self.num_zbins:
                self.ps_fid = torch.permute(self.ps_fid, (1, 0, 2, 3))
            self.ps_fid = self.ps_fid.reshape(self.num_spectra, self.num_zbins, self.num_kbins * self.num_ells)
        else:
            self.ps_fid = torch.zeros((self.num_spectra, self.num_zbins,  self.num_kbins * self.num_ells)).to(self.device)


    def _init_inverse_covariance(self):
        """Loads the inverse data covariance matrix for use in certain loss functions and normalizations"""

        # TODO: Upgrade to handle different number of k-bins for each zbin
        cov_file = self.input_dir+self.training_dir
        # Temporarily store with double percision to increase numerical stability\
        if os.path.exists(cov_file+"cov.dat"):
            cov = torch.load(cov_file+"cov.dat", weights_only=True).to(torch.float64)
        elif os.path.exists(cov_file+"cov.npy"):
            cov = torch.from_numpy(np.load(cov_file+"cov.npy"))
        else:
            print("Could not find covariance matrix! Using identity matrix instead...")
            cov = torch.eye(self.num_spectra*self.num_ells*self.num_kbins).unsqueeze(0)
            cov = cov.repeat(self.num_zbins, 1, 1)  

        self.invcov_blocks = get_invcov_blocks(cov, self.num_spectra, self.num_zbins, self.num_kbins, self.num_ells)
        self.invcov_full   = get_full_invcov(cov, self.num_zbins)


    def _diagonalize_covariance(self):
        """performs an eigenvalue decomposition of the each diagonal block of the inverse covariance matrix
           this function is always performed on cpu in double percision to improve stability"""
        
        self.Q = torch.zeros_like(self.invcov_blocks)
        self.Q_inv = torch.zeros_like(self.invcov_blocks)
        self.sqrt_eigvals = torch.zeros((self.num_spectra, self.num_zbins, self.num_ells*self.num_kbins))

        for (ps, z) in itertools.product(range(self.num_spectra), range(self.num_zbins)):
            eig, q = torch.linalg.eigh(self.invcov_blocks[ps, z])
            assert torch.all(torch.isnan(q)) == False
            assert torch.all(eig > 0), "ERROR! inverse covariance matrix has negative eigenvalues? Is it positive definite?"

            self.Q[ps, z] = q.real
            self.Q_inv[ps, z] = torch.linalg.inv(q).real
            self.sqrt_eigvals[ps, z] = torch.sqrt(eig)

        # move data to gpu and convert to single percision
        self.invcov_blocks = self.invcov_blocks.to(torch.float32).to(self.device)
        self.invcov_full = self.invcov_full.to(torch.float32).to(self.device)
        self.Q = self.Q.to(torch.float32).to(self.device)
        self.Q_inv = self.Q_inv.to(torch.float32).to(self.device)
        self.sqrt_eigvals = self.sqrt_eigvals.to(torch.float32).to(self.device)


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

    def _set_optimizer(self):
        """Sets optimization objects, one for each sub-network"""

        self.optimizer = [[[] for i in range(self.num_zbins)] for j in range(self.num_spectra)]
        self.scheduler = [[[] for i in range(self.num_zbins)] for j in range(self.num_spectra)]
        for (ps, z) in itertools.product(range(self.num_spectra), range(self.num_zbins)):
            net_idx = (z * self.num_spectra) + ps
            if self.optimizer_type == "Adam":
                self.optimizer[ps][z] = torch.optim.Adam(self.model.networks[net_idx].parameters(), 
                                                         lr=self.learning_rate)
            else:
                print("Error! Invalid optimizer type specified!")

            # use an adaptive learning rate
            self.scheduler[ps][z] = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer[ps][z],
                                    "min", factor=0.1, patience=15)

    def _save_model(self):
        """saves the current model state and normalization information to file"""

        # training statistics
        if not os.path.exists(self.input_dir+self.save_dir+"training_statistics"):
            os.mkdir(self.input_dir+self.save_dir+"training_statistics")

        for (ps, z) in itertools.product(range(self.num_spectra), range(self.num_zbins)):
            training_data = torch.vstack([torch.Tensor(self.train_loss[ps][z]), 
                                          torch.Tensor(self.valid_loss[ps][z]),
                                          torch.Tensor([self.train_time]*len(self.valid_loss[ps][z]))])
            torch.save(training_data, self.input_dir+self.save_dir+"training_statistics/train_data_"+str(ps)+"_"+str(z)+".dat")
        
        # configuration data
        # TODO Upgrade this to save more details (ex: what kbins was this trained with?)
        with open(self.input_dir+self.save_dir+'config.yaml', 'w') as outfile:
            yaml.dump(dict(self.config_dict), outfile, sort_keys=False, default_flow_style=False)

        # data related to input parameters
        # TODO: combine these into one file
        torch.save(self.input_normalizations, self.input_dir+self.save_dir+"input_normalizations.pt")
        with open(self.input_dir+self.save_dir+"param_names.txt", "w") as outfile:
            yaml.dump(self.get_required_parameters(), outfile, sort_keys=False, default_flow_style=False)

        # data related to output normalization
        output_files = [self.ps_fid, self.invcov_full, self.invcov_blocks, self.sqrt_eigvals, self.Q]
        torch.save(output_files, self.input_dir+self.save_dir+"output_normalizations.pt")
        # torch.save(self.ps_fid, self.input_dir+self.save_dir+"ps_fid.dat")
        # torch.save(self.invcov_full, self.input_dir+self.save_dir+"invcov_full.dat")
        # torch.save(self.invcov_blocks, self.input_dir+self.save_dir+"invcov_blocks.dat")
        # torch.save(self.sqrt_eigvals, self.input_dir+self.save_dir+"sqrt_eigenvals.dat")
        # torch.save(self.Q, self.input_dir+self.save_dir+"eigenvectors.dat")

        # Finally, the actual model parameters
        torch.save(self.model.state_dict(), self.input_dir+self.save_dir+'network.params')

    def _check_params(self, params):
        """checks that input parameters are in the expected format and within the specified boundaries"""

        if isinstance(params, torch.Tensor): params = params.to(self.device)
        else: params = torch.from_numpy(params).to(torch.float32).to(self.device)

        params = self.model.organize_parameters(params.unsqueeze(0))

        if torch.any(params < self.input_normalizations[0]) or \
           torch.any(params > self.input_normalizations[1]):
            print("WARNING: input parameters out of bounds! Emulator output will be untrustworthy:", params)
        
        norm_params = normalize_cosmo_params(params, self.input_normalizations)
        return norm_params

    def _train_one_epoch(self, train_loader, bin_idx):
        """basic training loop"""

        total_loss = 0.
        total_time = 0
        ps_idx = bin_idx[0]
        z_idx  = bin_idx[1]
        net_idx = (z_idx * self.num_spectra) + ps_idx
        for (i, batch) in enumerate(train_loader):
            t1 = time.time()
            
            # setup input parameters
            params = self.model.organize_parameters(batch[0])
            params = normalize_cosmo_params(params, self.input_normalizations)
            
            target = torch.flatten(batch[1][:,ps_idx,z_idx], start_dim=1)
            prediction = self.model.forward(params, net_idx)

            # calculate loss and update network parameters
            # TODO: Find better way to deal with passing invcov to the loss function
            loss = self.loss_function(prediction, target, self.invcov_full, True)
            assert torch.isnan(loss) == False
            assert torch.isinf(loss) == False
            self.optimizer[ps_idx][z_idx].zero_grad(set_to_none=True)
            loss.backward()

            self.optimizer[ps_idx][z_idx].step()
            total_loss += loss.detach()
            total_time += (time.time() - t1)

        if self.print_progress: print("time for epoch: {:0.1f}s, time per batch: {:0.1f}ms".format(total_time, 1000*total_time / len(train_loader)))
        return (total_loss / len(train_loader))