import torch
import numpy as np

from mentat_lss.utils import load_config_file, normalize_power_spectrum, un_normalize_power_spectrum

class pk_galaxy_dataset(torch.utils.data.Dataset):
    """Custom dataset storing large sets of galaxy power spectrum multipoles"""
    
    def __init__(self, data_dir:str, type:str, frac=1.):
        """Initializes the dataset.

        Args:
            data_dir (str): location to load the data to load
            type (str): One of ["training", "validation", "testing"]. Determines which data file to try loading.
            frac (float, optional): fraction of the full dataset to laod in. Defaults to 1..
        """
        self._load_data(data_dir, type, frac)


    def _load_data(self, data_dir:str, type:str, frac:float):
        """Loads in galaxy power spectrum multipoles from disk.

        Args:
            data_dir (str): location to load the data to load
            type (str): One of ["Training", "Validation", "Testing"]. Determines which data file to try loading.
            frac (float, optional): fraction of the full dataset to laod in. Defaults to 1..
        Raises:
            KeyError: If type is invalid.
        """
        if type.lower()=="training":
            file = data_dir+"pk-training.npz"
        elif type.lower()=="validation":
            file = data_dir+"pk-validation.npz"
        elif type.lower()=="testing":
            file = data_dir+"pk-testing.npz"
        else: 
            raise KeyError(f"Invalid dataset type! Must be [training, validation, testing], but got {type.lower()}")

        data = np.load(file)
        self.params = torch.from_numpy(data["params"]).to(torch.float32)
        self.galaxy_ps = torch.from_numpy(data["galaxy_ps"]).to(torch.float32)
        del data

        header_info = load_config_file(data_dir+"info.yaml")
        self.cosmo_params = header_info["cosmo_params"]
        self.bias_params = header_info["nuisance_params"]

        self.num_spectra = self.galaxy_ps.shape[1]
        self.num_zbins   = self.galaxy_ps.shape[2]
        self.num_kbins   = self.galaxy_ps.shape[3]
        self.num_ells    = self.galaxy_ps.shape[4]

        if frac != 1.:
            N_frac = int(self.params.shape[0] * frac)
            self.params = self.params[0:N_frac]
            self.galaxy_ps = self.galaxy_ps[0:N_frac]


    def __len__(self):
        """Returns the number of samples in the dataset

        Returns:
            len (int): number of samples in the dataset
        """
        return self.params.shape[0]
    

    def __getitem__(self, idx):
        """Returns specific items from the dataset

        Args:
            idx (int or torch.Tensor): index (or set od indices) to access

        Returns:
            params (torch.Tensor): input cosmology and bias parameters corresponding to idx
            galaxy_ps (torch.Tensor): (normalized) power spectrum multipoles corresponding to idx
            nw_ps (torch.Tensor): (normalized non-wiggle linear power spectra corresponding to idx. NOTE: in developement
            idx (int or torch.Tensor): The index of the corresponding data.
        """
        return self.params[idx], self.galaxy_ps[idx], idx


    def to(self, device:torch.device):
        """send data to the specified device, similar to the corresponding method for Tensors
        
        Args:
            device (torch.device): device to send the data to.
        """
        self.params = self.params.to(device)
        self.galaxy_ps = self.galaxy_ps.to(device)
    

    def normalize_data(self, ps_fid:torch.Tensor, sqrt_eigvals:torch.Tensor, Q:torch.Tensor):
        """Normalizes the reshapes the data

        Args:
            ps_fid (torch.Tensor): fiducial power spectrum multipoles in units of (Mpc/h)^3 used for normalization. Should have shape [nps, z, nk*nl]
            ps_nw_fid (torch.Tensor): NOTE: currently not used.
            sqrt_eigvals (torch.Tensor): set of sqrt eigenvalues used for normalization. Should have shape [ps, z, nk*nl]
            Q (torch.Tensor): set of eigenvectors used for normalization. Should have shape [ps, z, nk*nl, nk*nl]
        """
        self.galaxy_ps = normalize_power_spectrum(torch.flatten(self.galaxy_ps, start_dim=3), ps_fid, sqrt_eigvals, Q)
        self.galaxy_ps = self.galaxy_ps.reshape(-1, self.num_spectra, self.num_zbins, self.num_kbins*self.num_ells)

        #self.nw_ps = (self.nw_ps / ps_nw_fid) - 1.


    def get_normalized_galaxy_power_spectra(self, idx):
        """Returns the normalized power spectrum multipoles corresponding to idx

        Args:
            idx (int or torch.Tensor): index (or set of indexes) to access

        Returns:
            galaxy_ps[idx] (torch.Tensor): normalized power spectrum to access.
        """
        if isinstance(idx, int): 
            return torch.flatten(self.galaxy_ps[idx], start_dim=2)
        else:
            return torch.flatten(self.galaxy_ps[idx], start_dim=3)


    def get_true_galaxy_power_spectra(self, idx, ps_fid:torch.Tensor, sqrt_eigvals:torch.Tensor, Q:torch.Tensor, Q_inv:torch.Tensor):
        """Returns the galaxy power spectrum multipoles in units of (Mpc/h)^3 corresponding to idx

        Args:
            idx (int or torch.Tensor): index (or set of indexes) to access
            ps_fid (torch.Tensor): fiducial power spectrum used to reverse normalization. Expected shape is [nps*nz, nk*nl]  
            sqrt_eigvals (torch.Tensor): square root eigenvalues of the inverse covariance matrix. Expected shape is [nps*nz, nk*nl]  
            Q (torch.Tensor): eigenvectors of the inverse covariance matrix. Expected shape is [nps*nz, nk*nl, nk*nl]  
            Q_inv (torch.Tensor): inverse eigenvectors of the inverse covariance matrix. Expected shape is [nps*nz, nk*nl, nk*nl]  

        Returns:
           galaxy_ps[idx] (torch.Tensor): galaxy power spectrum in units of (Mpc/h)^3 to access. has shape [b, nps, nz, nk, nl] or [nps, nz, nk, nl]
        """
        if isinstance(idx, int): 
            flatten_dim = 2
            final_shape = (self.num_spectra, self.num_zbins, self.num_kbins, self.num_ells)
        else:          
            flatten_dim = 3
            final_shape = (-1, self.num_spectra, self.num_zbins, self.num_kbins, self.num_ells)

        ps_true = un_normalize_power_spectrum(torch.flatten(self.galaxy_ps[idx], start_dim=flatten_dim), ps_fid, sqrt_eigvals, Q, Q_inv)
        ps_true = ps_true.reshape(final_shape)        
        return ps_true