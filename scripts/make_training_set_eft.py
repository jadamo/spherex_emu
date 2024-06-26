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

k = np.linspace(0.01, 0.25, 25)

N = 500
#N_PROC=int(os.environ["SLURM_CPUS_ON_NODE"])
N_PROC=14

# fraction of dataset to be partitioned to the training | validation | test sets
train_frac = 0.8
valid_frac = 0.1
test_frac  = 0.1

config_dir = filepaths.network_pars_dir+"network_pars_single_tracer_2_redshift.yaml"
save_dir = "/home/joeadamo/Research/Data/SPHEREx-Data/Training-Set-EFT-1t-2z/"

#-------------------------------------------------------------------
# FUNCTIONS
#-------------------------------------------------------------------

def prepare_header_info(param_names, fiducial_cosmology, n_samples):

    header_info = {}
    header_info["total_samples"] = n_samples
    cosmo_params = []
    bias_params = []
    for key in fiducial_cosmology["cosmo_params"].keys():
        if key in param_names:
            cosmo_params.append(key)
    for key in fiducial_cosmology["bias_params"].keys():
        if key in param_names:
            bias_params.append(key)

    header_info["cosmo_params"] = sorted(cosmo_params)
    header_info["bias_params"] = bias_params
    return header_info

def get_power_spectrum(sample, param_names, fiducial_cosmology, ps_config):

    ells = [0, 2]
    num_tracers = 1
    num_zbins = len(ps_config["redshift_list"])
    sample_dict = dict(zip(param_names, sample))
    param_vector = prepare_ps_inputs(sample_dict, fiducial_cosmology, num_tracers, num_zbins)

    try:
        theory = ps_theory_calculator.PowerSpectrumMultipole1Loop(ps_config)
        galaxy_ps = theory(k, ells, param_vector)

        if not np.any(np.isnan(galaxy_ps)): return galaxy_ps, 0
        else: return np.zeros(2, len(k)), -1
    except:
        print("Power spectrum calculation failed!")
        return np.zeros(2, len(k)), -1

#-------------------------------------------------------------------
# MAIN
#-------------------------------------------------------------------
def main():
    
    assert os.path.exists(save_dir)

    fiducial_cosmology = load_config_file(filepaths.cosmo_pars_dir+"eft_single_tracer_fiducial.yaml")
    survey_pars = load_config_file(filepaths.survey_pars_dir+'survey_pars_single_tracer_2_redshift.yaml')
    config_dict = load_config_file(config_dir)

    param_names, param_ranges = get_parameter_ranges(config_dict)
    samples = make_latin_hypercube(param_ranges, N)
    ndens_table = np.array([[float(survey_pars['number_density_in_hinvMpc_%s' % (i+1)][j]) for j in range(survey_pars['nz'])] for i in range(survey_pars['nsample'])])
    z_eff = (np.array(survey_pars["zbin_lo"]) + np.array(survey_pars["zbin_hi"])) / 2.
    
    # table of number densities of tracer samples. 
    # (i, j) component is the number density of the i-th sample at the j-th redshift.
    ps_config = {}
    ps_config['number_density_table'] = ndens_table
    ps_config['redshift_list'] = z_eff # redshift bins
    ps_config['Omega_m_ref'] = 0.3 # Omega_m value of the reference cosmology (assuming a flat LambdaCDM)

    print("Generating", str(N), "power spectra with", str(N_PROC), "processors...")
    print("Number of samples:", ps_config['number_density_table'].shape[0])
    print("Number of redshift bins:", len(ps_config["redshift_list"]))
    print("Saving to", save_dir)

    # initialize pool for multiprocessing
    t1 = time.time()
    p = Pool(processes=N_PROC)
    pk, result = zip(*p.starmap(get_power_spectrum, 
                      zip(samples, repeat(param_names), repeat(fiducial_cosmology), repeat(ps_config))))
    p.close()
    p.join()

    # aggregate data
    pk = np.array(pk)
    result = np.array(result)

    idx_pass = np.where(result == 0)[0]
    fail_compute = len(np.where(result == -1)[0])

    pk = pk[idx_pass]
    samples = samples[idx_pass]

    dataset_info = prepare_header_info(param_names, fiducial_cosmology, N - fail_compute)

    if pk.shape[0] > 1:
        np.savez(save_dir+"pk-raw.npz", params=samples, pk=pk)
        with open(save_dir+'info.yaml', 'w') as outfile:
            yaml.dump(dataset_info, outfile, sort_keys=False, default_flow_style=False)

        # F_test = np.load(save_dir+"pk-raw.npz")
        # print(F_test["params"].shape, F_test["pk"].shape)
    t2 = time.time()
    print("Done! Took {:0.0f} hours {:0.0f} minutes".format(math.floor((t2 - t1)/3600), math.floor((t2 - t1)/60%60)))
    print("Made {:0.0f} / {:0.0f} galaxy power spectra".format(N - fail_compute, N))
    print("{:0.0f} ({:0.2f}%) power spectra failed to compute".format(fail_compute, 100.*fail_compute / N))

    organize_training_set(save_dir, train_frac, valid_frac, test_frac,
                          samples.shape[1], len(z_eff), ps_config['number_density_table'].shape[0], len(k), True)

if __name__ == "__main__":
    main()
