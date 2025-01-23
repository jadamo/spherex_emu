import numpy as np
import torch
import ultranest
import ultranest.stepsampler
import time

import ps_theory_calculator
from spherex_emu.emulator import pk_emulator
import spherex_emu.filepaths as filepaths
from spherex_emu.utils import load_config_file, get_parameter_ranges, prepare_ps_inputs, make_latin_hypercube

# Defining "global" things here
# TODO: find better way to do this

# load in the fiducial data vector, emulator, and (inverse) covariance matrix
emulator_dir = filepaths.base_dir+"/emulators/transformer_3_sample_1_redshift/"
training_dir = filepaths.base_dir+"../../Data/SPHEREx-Data/Training-Set-EFT-3s-1z/"
save_dir = filepaths.base_dir+"chains/eft_3_tracer/"
invcov_dir = "/home/joeadamo/Research/SPHEREx/covapt_mt/data/output_data/invcov.dat"
use_emulator = False

# get cosmology and survey parameters
cosmo_dict = load_config_file(filepaths.cosmo_pars_dir+"eft_3_sample_1_redshift.yaml")
survey_pars = load_config_file(filepaths.survey_pars_dir + 'survey_pars_3_sample_1_redshift.yaml')

invcov_data = torch.load(invcov_dir)
invcov = invcov_data["zbin_0"].to("cpu").detach().numpy()

config_dict = load_config_file(emulator_dir+"config.yaml")
config_dict["use_gpu"] = False

k_data = np.load(training_dir+"kbins.npz")
k = k_data["k_0"]
ells = [0,2]

emulator = pk_emulator(emulator_dir+"config.yaml")
emulator.load_trained_model()

data_vector_file = "/home/joeadamo/Research/SPHEREx/covapt_mt/data/input_data/ps_emu_test_3_tracer_no_noise.npy"
data_vector = np.load(data_vector_file)
#data_vector = emulator.ps_fid.view(emulator.num_zbins, emulator.num_spectra, emulator.num_kbins, 2).to("cpu").detach().numpy()
#data_vector = data_vector.transpose(0, 1, 3, 2)
print(data_vector.shape)

# ps_1loop setup
ndens_table = np.array([[float(survey_pars['number_density_in_hinvMpc_%s' % (i+1)][j]) for j in range(survey_pars['nz'])] for i in range(survey_pars['nsample'])])
z_eff = (np.array(survey_pars["zbin_lo"]) + np.array(survey_pars["zbin_hi"])) / 2.

# table of number densities of tracer samples. 
# (i, j) component is the number density of the i-th sample at the j-th redshift.
ps_config = {}
ps_config['number_density_table'] = ndens_table
ps_config['redshift_list'] = list(z_eff) # redshift bins
ps_config['Omega_m_ref'] = 0.3 # Omega_m value of the reference cosmology (assuming a flat LambdaCDM)
theory = ps_theory_calculator.PowerSpectrumMultipole1Loop(ps_config)

param_names, param_ranges = get_parameter_ranges(cosmo_dict)
#param_ranges = param_ranges[:2]
#param_names = param_names[:2]

# manually define parameters to vary
# param_names = ["fnl", "b1", "b2"]
# param_ranges = np.array([[-75, 75],
#                          [0.5, 4.0],
#                          [-1.5, 0.5]])

# define the likelhood functions
def prior(theta, param_ranges):
    is_too_low = bool(np.sum(theta < param_ranges[:,0]))
    is_too_high = bool(np.sum(theta > param_ranges[:,1]))
    if is_too_low or is_too_high: return -np.inf
    else:                         return 0

def prior_transform(cube):
    params = cube * (param_ranges[:,1] - param_ranges[:,0]) + param_ranges[:,0]
    return params

def get_eft_model_vector(theta):
    params_dict = {}
    idx = 0
    for pname in list(param_names):
        params_dict[pname] = theta[idx]
        idx+=1

    params = prepare_ps_inputs(params_dict, cosmo_dict, 3, 1)
    # If varying h, make sure to divide by the varied h value!
    model_vector = theory(k, ells, params) / cosmo_dict["cosmo_params"]["h"]["value"]**3
    #model_vector = theory(k, ells, params) / params_dict["h"]**3
    return model_vector

def log_lkl(theta):

    params = theta
    #params = np.concatenate((theta, list(cosmo_fid.values())[9:]))
    # nz, nps, nl, nk
    if use_emulator: model_vector = emulator.get_power_spectra(params)
    else:            model_vector = get_eft_model_vector(params)

    delta = data_vector - model_vector # (nz, nps, nl, nk)
    delta = delta.transpose(0, 1, 3, 2) # (b,nz,nps,nl,nk) -> (b, nz, nps, nk, nl)
    (nz, nps, nk, nl) = delta.shape
    delta = delta.reshape((nz, nps*nk*nl)) # (nz, nps, nk, nl) --> (nz, nps*nk*nl) 
    delta_row = delta[:, None, :,] # (nz, 1, nps*nk*nl) 
    delta_col = delta[:, :, None,] # (nz, nps*nk*nl, 1) 

    # NOTE Matrix multiplication is for the last two indices; element wise for all other indices.
    chi2_component = np.matmul(delta_row, np.matmul(invcov, delta_col))[..., 0, 0] # invcov is (nz, nps*nk*nl, nps*nk*nl)
    chi2 = -0.5 * np.sum(chi2_component)
    assert chi2 < 0, "ERROR: chi2 value is negative!"
    return chi2 + prior(theta, param_ranges)

def main():

    print("Varied params :", param_names)
    print("Saving chain to: " + save_dir)

    sampler = ultranest.ReactiveNestedSampler(param_names, log_lkl, prior_transform,
                                        log_dir=save_dir, resume="overwrite")
    sampler.stepsampler = ultranest.stepsampler.SliceSampler(
       nsteps=40,
       generate_direction=ultranest.stepsampler.generate_mixture_random_direction,
        adaptive_nsteps=False,
        max_nsteps=400
    )
    t1 = time.time()
    results = sampler.run(update_interval_volume_fraction=0.7,
                          max_ncalls=1e6)
    t2 = time.time()
    sampler.print_results()
    print("Done in {:0.0f}m {:0.1f}s".format(int((t2-t1) / 60), (t2-t1) % 60))

if __name__ == "__main__":
    main()