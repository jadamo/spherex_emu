import torch
#import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import os

from spherex_emu.utils import normalize, un_normalize, load_config_file

class pk_galaxy_dataset(torch.utils.data.Dataset):

    def __init__(self, data_dir:str, type:str, frac=1.):
        
        self._load_data(data_dir, type, frac)
        self._set_normalization(data_dir, type)
        self.pk = normalize(self.pk, self.output_normalizations)
        self.pk = self.pk.view(-1, self.num_zbins * self.num_samples, self.num_ells * self.num_kbins)

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

        self.num_zbins = self.pk.shape[1]
        self.num_samples = self.pk.shape[2]
        self.num_ells = self.pk.shape[3]
        self.num_kbins = self.pk.shape[4]

        if frac != 1.:
            N_frac = int(self.params.shape[0] * frac)
            self.params = self.params[0:N_frac]
            self.pk = self.pk[0:N_frac]

    def _set_normalization(self, data_dir, type):
        """finds the min and max values for each multipole and saves to another file"""
        self.output_normalizations = torch.zeros(2, self.num_zbins, self.num_samples, self.num_ells, 1)
        if type == "training":
            for zbin in range(self.num_zbins):
                for sample in range(self.num_samples):
                    for ell in range(self.num_ells):
                        self.output_normalizations[0,zbin,sample,ell] = torch.amin(self.pk[:,zbin, sample, ell, :]).item()
                        self.output_normalizations[1,zbin,sample,ell] = torch.amax(self.pk[:,zbin, sample, ell, :]).item()
            torch.save(self.output_normalizations, data_dir+"pk-normalization.dat")
        elif os.path.exists(data_dir+"pk-normalization.dat"):
            self.output_normalizations = torch.load(data_dir+"pk-normalization.dat")

    def __len__(self):
        return self.params.shape[0]
    
    def __getitem__(self, idx):
        return self.params[idx], self.pk[idx], idx

    def to(self, device):
        """send data to the specified device"""
        self.params = self.params.to(device)
        self.pk = self.pk.to(device)

    def get_repeat_params(self, idx, num_redshift, num_samples):
        """returns a 3D or 4D tensor of separated parameters for each sample and redshift bin"""        
        if num_samples == 1 and num_redshift == 1: return self.params[idx]

        return_params = self.params[idx]
        if isinstance(idx, int): 
            return_params = return_params.unsqueeze(0).repeat(num_redshift, 1)
            return_params = return_params.unsqueeze(1).repeat(1, num_samples, 1)
        else:
            return_params = return_params.unsqueeze(1).repeat(1, num_redshift, 1)
            return_params = return_params.unsqueeze(2).repeat(1, 1, num_samples, 1)

        return return_params

    def get_norm_values(self):
        return self.normalizations

    def get_power_spectra(self, idx):
        
        pk = self.pk[idx].view(self.num_zbins, self.num_samples, self.num_ells, self.num_kbins)
        pk = un_normalize(pk, self.normalizations).detach().numpy()
        return pk