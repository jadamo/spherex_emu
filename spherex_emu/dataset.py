import torch
#import torch.nn as nn
import numpy as np

from spherex_emu.utils import load_config_file

class pk_galaxy_dataset(torch.utils.data.Dataset):

    def __init__(self, data_dir:str, type:str, frac=1.):
        
        self._load_data(data_dir, type, frac)
        #self._set_normalization(data_dir, type)
        #self.pk = normalize(self.pk, self.output_normalizations)
        #self.pk = self.pk.view(-1, self.num_zbins * self.num_samples, self.num_ells * self.num_kbins)
        self.pk = self.pk.reshape(-1, self.num_spectra, self.num_zbins, self.num_kbins*self.num_ells)

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
    
    def get_power_spectra(self, idx):        
        return self.pk[idx].view(self.num_spectra, self.num_zbins, self.num_kbins, self.num_ells)

