# This script generates a training set for the spherex emulator based on Yosuke's EFT power spectrum model
# Utilizing an embarrasingly parallel code structure

import time, math, yaml
import numpy as np

from multiprocessing import Pool
from itertools import repeat

import ps_theory_calculator
from spherex_emu.utils import *
import spherex_emu.filepaths as filepaths 

#-------------------------------------------------------------------
# GLOBAL VARIABLES
#-------------------------------------------------------------------

k = np.linspace(0.07, 0.2, 50)

N = 100000

# fraction of dataset to be partitioned to the training | validation | test sets
train_frac = 0.8
valid_frac = 0.1
test_frac  = 0.1

net_config_file = filepaths.network_pars_dir+"network_pars_single_sample_single_redshift.yaml"
cosmo_config_file = filepaths.cosmo_pars_dir+"eft_single_sample_single_redshift.yaml"
survey_config_file = filepaths.survey_pars_dir+'survey_pars_single_sample_single_redshift.yaml'

#Joes directory: save_dir = "/home/joeadamo/Research/Data/SPHEREx-Data/Training-Set-EFT-2s-2z/"

#Same filepath to save as tns training set
#save_dir = filepaths.data_dir
save_dir = '/home/u12/jadamo/Data/Training-Set-EFT-1s-1z/'

#-------------------------------------------------------------------
# MAIN
#-------------------------------------------------------------------
def main():
    
    assert os.path.exists(save_dir)

    cosmo_dict  = load_config_file(cosmo_config_file)
    survey_dict = load_config_file(survey_config_file)
    config_dict = load_config_file(net_config_file)

    ndens_table = np.array([[float(survey_dict['number_density_in_hinvMpc_%s' % (i+1)][j]) for j in range(survey_dict['nz'])] for i in range(survey_dict['nsample'])])
    z_eff = (np.array(survey_dict["zbin_lo"]) + np.array(survey_dict["zbin_hi"])) / 2.
    num_spectra = ndens_table.shape[0] + math.comb(ndens_table.shape[0], 2)
    num_ells = 5

    # table of number densities of tracer samples. 
    # (i, j) component is the number density of the i-th sample at the j-th redshift.
    ps_config = {}
    ps_config['number_density_table'] = ndens_table
    ps_config['redshift_list'] = z_eff # redshift bins
    ps_config['Omega_m_ref'] = 0.3 # Omega_m value of the reference cosmology (assuming a flat LambdaCDM)

    param_names, param_ranges = get_parameter_ranges(cosmo_dict)

    organize_training_set(save_dir, train_frac, valid_frac, test_frac,
                          3, len(z_eff), num_spectra, num_ells, len(k), False)

if __name__ == "__main__":
    main()
