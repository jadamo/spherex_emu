import torch
#import torch.nn as nn
import numpy as np

from spherex_emu.utils import load_config_file, normalize_power_spectrum, un_normalize_power_spectrum

class pk_galaxy_dataset(torch.utils.data.Dataset):

    def __init__(self, data_dir:str, type:str, frac=1.):
        
        self._load_data(data_dir, type, frac)

        #self.pk = self.pk.reshape(-1, self.num_zbins, self.num_spectra, self.num_kbins, self.num_ells)

    def _load_data(self, data_dir, type, frac):

        if type=="training":
            file = data_dir+"pk-training.npz"
        elif type=="validation":
            file = data_dir+"pk-validation.npz"
        elif type=="testing":
            file = data_dir+"pk-testing.npz"
        else: print("ERROR! Invalid dataset type! Must be [training, validation, testing]")

        data = np.load(file)
        self.params = torch.from_numpy(data["params"]).to(torch.float32)
        self.pk = torch.from_numpy(data["pk"]).to(torch.float32)
        del data

        header_info = load_config_file(data_dir+"info.yaml")
        self.cosmo_params = header_info["cosmo_params"]
        self.bias_params = header_info["bias_params"]

        # TODO: change training set script such that we can remove this line
        self.pk = torch.permute(self.pk, (0, 2, 1, 4, 3))

        self.num_spectra = self.pk.shape[1]
        self.num_zbins   = self.pk.shape[2]
        self.num_kbins   = self.pk.shape[3]
        self.num_ells    = self.pk.shape[4]

        if frac != 1.:
            N_frac = int(self.params.shape[0] * frac)
            self.params = self.params[0:N_frac]
            self.pk = self.pk[0:N_frac]

    def __len__(self):
        return self.params.shape[0]
    
    def __getitem__(self, idx):
        return self.params[idx], self.pk[idx], idx

    def to(self, device):
        """send data to the specified device"""
        self.params = self.params.to(device)
        self.pk = self.pk.to(device)
    
    def normalize_data(self, ps_fid, sqrt_eigvals, Q):
        self.pk = normalize_power_spectrum(torch.flatten(self.pk, start_dim=3), ps_fid, sqrt_eigvals, Q)
        self.pk = self.pk.reshape(-1, self.num_spectra, self.num_zbins, self.num_kbins*self.num_ells)

    def get_normalized_power_spectra(self, idx):
        
        if isinstance(idx, int): 
            return torch.flatten(self.pk[idx], start_dim=2)
        else:
            return torch.flatten(self.pk[idx], start_dim=3)
    
    def get_true_power_spectra(self, idx, ps_fid, sqrt_eigvals, Q, Q_inv):
        
        if isinstance(idx, int): 
            flatten_dim = 2
            final_shape = (self.num_spectra, self.num_zbins, self.num_kbins, self.num_ells)
        else:                    
            flatten_dim = 3
            final_shape = (-1, self.num_spectra, self.num_zbins, self.num_kbins, self.num_ells)

        ps_true = un_normalize_power_spectrum(torch.flatten(self.pk[idx], start_dim=flatten_dim), ps_fid, sqrt_eigvals, Q, Q_inv)
        ps_true = ps_true.reshape(final_shape)        
        return ps_true
