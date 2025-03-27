# This script generates a training set for the spherex emulator based on Yosuke's EFT power spectrum model
# Utilizing an embarrasingly parallel code structure

import time, math, yaml, sys
import numpy as np

from multiprocessing import Pool
from itertools import repeat

import ps_theory_calculator
from spherex_emu.utils import *

#-------------------------------------------------------------------
# GLOBAL VARIABLES
#-------------------------------------------------------------------

N = 100000
N_PROC=int(os.environ["SLURM_CPUS_ON_NODE"])
#N_PROC=14

# fraction of dataset to be partitioned to the training | validation | test sets
train_frac = 0.8
valid_frac = 0.1
test_frac  = 0.1

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
    for key in fiducial_cosmology["nuisance_params"].keys():
        if key in param_names:
            bias_params.append(key)

    header_info["cosmo_params"] = cosmo_params
    header_info["nuisance_params"] = bias_params
    return header_info

def get_power_spectrum(sample, k, param_names, cosmo_dict, ps_config):

    ells = [0, 2]
    num_samples = ps_config['number_density_table'].shape[0]
    num_zbins = len(ps_config["redshift_list"])
    sample_dict = dict(zip(param_names, sample))

    param_vector = prepare_ps_inputs(sample_dict, cosmo_dict, num_samples, num_zbins)
    try:
        theory = ps_theory_calculator.PowerSpectrumMultipole1Loop(ps_config)
        galaxy_ps = theory(k, ells, param_vector) / cosmo_dict["cosmo_params"]["h"]["value"]**3

        if not np.any(np.isnan(galaxy_ps)): return galaxy_ps, 0
        else: return np.zeros(len(ells), len(k)), -1
    except:
        print("Power spectrum calculation failed!")
        return np.zeros(len(ells), len(k)), -1

#-------------------------------------------------------------------
# MAIN
#-------------------------------------------------------------------
def main():

    if len(sys.argv) < 5:
        print("USAGE: python make_training_set_eft.py <cosmo_config_file> <survey_config_file> <save_dir> <k array file>")
        return 0

    cosmo_config_file  = sys.argv[1]
    survey_config_file = sys.argv[2]
    save_dir           = sys.argv[3]
    k_array_file       = sys.argv[4]

    if not os.path.exists(save_dir):
        print("Attempting to create save directory...")
        os.mkdir(save_dir)

    cosmo_dict  = load_config_file(cosmo_config_file)
    survey_dict = load_config_file(survey_config_file)

    # TODO: Upgrade to handle different k-bins for different redshifts
    k_data = np.load(k_array_file)
    k = k_data["k"]

    ndens_table = np.array([[float(survey_dict['number_density_in_hinvMpc_%s' % (i+1)][j]) for j in range(survey_dict['nz'])] for i in range(survey_dict['nsample'])])
    z_eff = (np.array(survey_dict["zbin_lo"]) + np.array(survey_dict["zbin_hi"])) / 2.
    num_spectra = ndens_table.shape[0] + math.comb(ndens_table.shape[0], 2)

    # table of number densities of tracer samples. 
    # (i, j) component is the number density of the i-th sample at the j-th redshift.
    ps_config = {}
    ps_config['number_density_table'] = ndens_table
    ps_config['redshift_list'] = z_eff # redshift bins
    ps_config['Omega_m_ref'] = 0.3 # Omega_m value of the reference cosmology (assuming a flat LambdaCDM)

    param_names, param_ranges = get_parameter_ranges(cosmo_dict)
    samples = make_latin_hypercube(param_ranges, N)

    print("Generating", str(N), "power spectra with", str(N_PROC), "processors...")
    print("Number of samples:", ps_config['number_density_table'].shape[0])
    print("Number of redshift bins:", len(ps_config["redshift_list"]))
    print("Number of varied parameters:", len(param_names),':', param_names)
    print("Saving to", save_dir)

    # first, generate the power spectrum at the fiducial cosmology
    print("Generating fiducial power spectrum...")
    pk, result = get_power_spectrum({}, param_names, cosmo_dict, ps_config)
    if result == 0:
        np.save(save_dir+"ps_fid.npy", pk)
        np.save(save_dir+"kbins.npz", k=k)
    else:
        print("ERROR! failed to calculate fiducial power spectrum! Exiting...")
        return -1

    # # initialize pool for multiprocessing
    t1 = time.time()
    print("Generating training set...")
    p = Pool(processes=N_PROC)
    pk, result = zip(*p.starmap(get_power_spectrum, 
                      zip(samples, repeat(k), repeat(param_names), repeat(cosmo_dict), repeat(ps_config))))
    p.close()
    p.join()

    # aggregate data
    pk = np.array(pk)
    result = np.array(result)

    idx_pass = np.where(result == 0)[0]
    fail_compute = len(np.where(result == -1)[0])

    pk = pk[idx_pass]
    samples = samples[idx_pass]

    dataset_info = prepare_header_info(param_names, cosmo_dict, N - fail_compute)

    if pk.shape[0] > 1:
        np.savez(save_dir+"pk-raw.npz", params=samples, pk=pk)
        with open(save_dir+'info.yaml', 'w') as outfile:
            yaml.dump(dataset_info, outfile, sort_keys=False, default_flow_style=False)

    del pk

    t2 = time.time()
    print("Done! Took {:0.0f} hours {:0.0f} minutes".format(math.floor((t2 - t1)/3600), math.floor((t2 - t1)/60%60)))
    print("Made {:0.0f} / {:0.0f} galaxy power spectra".format(N - fail_compute, N))
    print("{:0.0f} ({:0.2f}%) power spectra failed to compute".format(fail_compute, 100.*fail_compute / N))

    organize_training_set(save_dir, train_frac, valid_frac, test_frac,
                          samples.shape[1], len(z_eff), num_spectra, 2, len(k), True)

if __name__ == "__main__":
    main()
