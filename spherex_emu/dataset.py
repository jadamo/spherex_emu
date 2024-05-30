import torch
#import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import os

from spherex_emu.utils import normalize, un_normalize

class pk_galaxy_dataset(torch.utils.data.Dataset):

    def __init__(self, data_dir:str, type:str, frac=1.):
        
        self._load_data(data_dir, type, frac)
        self._set_normalization(data_dir, type)
        self.pk = normalize(self.pk, self.normalizations)
        self.pk = self.pk.view(-1, self.num_zbins * self.num_tracers * self.num_ells * self.num_kbins)

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
        
        self.num_zbins = self.pk.shape[1]
        self.num_tracers = self.pk.shape[2]
        self.num_ells = self.pk.shape[3]
        self.num_kbins = self.pk.shape[4]

        if frac != 1.:
            N_frac = int(self.params.shape[0] * frac)
            self.params = self.params[0:N_frac]
            self.pk = self.pk[0:N_frac]

    def _set_normalization(self, data_dir, type):
        """finds the min and max values for each multipole and saves to another file"""
        self.normalizations = torch.zeros(2, self.num_tracers, self.num_zbins, self.num_ells, 1)
        if type == "training":
            for tracer in range(self.num_tracers):
                for zbin in range(self.num_zbins):
                    for ell in range(self.num_ells):
                        self.normalizations[0,tracer,zbin,ell] = torch.amin(self.pk[:,tracer, zbin, ell, :]).item()
                        self.normalizations[1,tracer,zbin,ell] = torch.amax(self.pk[:,tracer, zbin, ell, :]).item()
            torch.save(self.normalizations, data_dir+"pk-normalization.dat")
        elif os.path.exists(data_dir+"pk-normalization.dat"):
            self.normalizations = torch.load(data_dir+"pk-normalization.dat")

    def __len__(self):
        return self.params.shape[0]
    
    def __getitem__(self, idx):
        return self.params[idx], self.pk[idx]

    def to(self, device):
        """send data to the specified device"""
        self.params = self.params.to(device)
        self.pk = self.pk.to(device)

    def get_norm_values(self):
        return self.normalizations

    def get_power_spectra(self, idx):
        
        pk = self.pk[idx].view(self.num_zbins, self.num_tracers, self.num_ells, self.num_kbins)
        pk = un_normalize(pk, self.normalizations).detach().numpy()
        return pk