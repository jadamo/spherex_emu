# This script generates a training set for the spherex emulator based on Yosuke's EFT power spectrum model
# Utilizing an embarrasingly parallel code structure

import time, math, yaml, itertools
import camb
import numpy as np

from multiprocessing import Pool
from itertools import repeat

import ps_theory_calculator
from mentat_lss.gpsclass import CalcGalaxyPowerSpec
from mentat_lss.utils import *

#-------------------------------------------------------------------
# GLOBAL VARIABLES
#-------------------------------------------------------------------

N = 10
N_PROC=int(os.environ["SLURM_CPUS_ON_NODE"])

# TODO: Place this info in a better location like a survey pars file
kmin = 0.01
kmax = 0.25
kbins = 26 # smallest possible is 4 and has to be even

# fraction of dataset to be partitioned to the training | validation | test sets
train_frac = 0.8
valid_frac = 0.1
test_frac  = 0.1

net_config_file = filepaths.network_pars_dir+"network_pars_2_sample_2_redshift.yaml"
cosmo_config_file = filepaths.cosmo_pars_dir+"tns_2_sample_2_redshift.yaml"
survey_config_file = filepaths.survey_pars_dir+'survey_pars_2_sample_2_redshift.yaml'

save_dir = "/Users/JoeyA/Research/Data/SPHEREx-Data/Training-Set-TNS-2s-2z/"

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

#Creates linear power spectra from priors - input into galaxy ps class
def get_power_spectrum(sample, param_names, cosmo_dict, ps_config):

    num_sample_bins = ps_config['number_density_table'].shape[0]
    num_zbins = len(ps_config["redshift_list"])
    sample_dict = dict(zip(param_names, sample))

    param_vector = prepare_ps_inputs(sample_dict, cosmo_dict, num_sample_bins, num_zbins)

    param_vector[0] = param_vector[0] * 100
    num_spectra = num_sample_bins + math.comb(num_sample_bins, 2)
    
    num_cosmo_params = 5
    num_bias_params = 4
    galaxy_ps = np.zeros((len(ps_config['redshift_list']), num_spectra, 2, kbins))
    k = np.zeros((kbins)) #number of samples x number of k bins
    
    try:
        for z_idx in range(len(ps_config['redshift_list'])):
            z = ps_config['redshift_list'][z_idx]
            H0, ombh2, omch2, As, ns = param_vector[0], param_vector[1], param_vector[2], param_vector[3], param_vector[4]
            pars = camb.CAMBparams()
            pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2)
            pars.InitPower.set_params(As=As, ns=ns)
            pars.set_matter_power(redshifts=[z], kmax=2.0) #sets redshift and mode for ps
            pars.NonLinear = camb.model.NonLinear_none #set to be linear
            results = camb.get_results(pars)
            kh, z_camb, pk_lin = results.get_matter_power_spectrum(minkh=kmin, maxkh=kmax, npoints=kbins) #pk is 2 values 
            
            Om0 = results.get_Omega("cdm", 0) + results.get_Omega("baryon", 0)
            f = fgrowth(z, Om0)
            k = kh

            sample_idx = 0
            for isample1, isample2 in itertools.product(range(num_sample_bins), repeat=2):
                if isample1 > isample2: continue

                idx1 = num_cosmo_params + (isample1*num_bias_params) + (z_idx * num_bias_params*num_sample_bins)
                idx2 = num_cosmo_params + (isample2*num_bias_params) + (z_idx * num_bias_params*num_sample_bins)
                bias1 = param_vector[idx1:idx1+num_bias_params]
                bias2 = param_vector[idx2:idx2+num_bias_params]

                nonlin = CalcGalaxyPowerSpec(f,pk_lin[0],kh,bias1,bias2,param_vector[:num_cosmo_params])
                galaxy_ps[z_idx, sample_idx, 0, :] = nonlin.get_nonlinear_ps(0)
                galaxy_ps[z_idx, sample_idx, 1, :] = nonlin.get_nonlinear_ps(2)
                sample_idx+=1

        if not np.any(np.isnan(galaxy_ps)): return galaxy_ps, 0
        else: return np.zeros(num_zbins, num_spectra, 2, kbins), -1
    except:
        print("Power spectrum calculation failed!")
        return np.zeros(num_zbins, num_spectra, 2, kbins), -1

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
    print("Number of varied parameters:", len(param_names))
    print("Saving to", save_dir)

    # # initialize pool for multiprocessing
    t1 = time.time()
    p = Pool(processes=N_PROC)
    pk, result = zip(*p.starmap(get_power_spectrum, 
                      zip(samples, repeat(param_names), repeat(cosmo_dict), repeat(ps_config))))
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
                          samples.shape[1], len(z_eff), num_spectra, kbins, True)

if __name__ == "__main__":
    main()
