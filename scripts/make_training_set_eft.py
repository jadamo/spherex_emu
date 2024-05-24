# This script is basically CovaPT's jupyter notebook, but in script form so you can more easily run it

import time, os, warnings, math, yaml
import numpy as np

from multiprocessing import Pool
from itertools import repeat
#from mpi4py import MPI

import ps_theory_calculator
from spherex_emu.utils import *
from spherex_emu.filepaths import net_config_dir

#-------------------------------------------------------------------
# GLOBAL VARIABLES
#-------------------------------------------------------------------

survey_pars_file = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
#survey_pars_file += '/configs/survey_pars_v28_base_cbe.yaml'
survey_pars_file += '/configs/survey_pars_single_tracer_single_redshift_bin.yaml'
with open(survey_pars_file) as file:
    survey_pars = yaml.load(file, Loader=yaml.FullLoader)
ndens_table = np.array([[float(survey_pars['number_density_in_hinvMpc_%s' % (i+1)][j]) for j in range(survey_pars['nz'])] for i in range(survey_pars['nsample'])])

# table of number densities of tracer samples. 
# (i, j) component is the number density of the i-th sample at the j-th redshift.
ps_config = {}
ps_config['number_density_table'] = ndens_table
ps_config['redshift_list'] = [0.1, 0.3, 0.5, 0.7, 0.9, 1.3, 1.9, 2.5, 3.1, 3.7, 4.3] # redshift bins
ps_config['Omega_m_ref'] = 0.3 # Omega_m value of the reference cosmology (assuming a flat LambdaCDM)

cosmo_list = ['h', 'omega_b', 'omega_c', 'As', 'ns', 'fnl']
bias_list = ['b1', 'b2', 'bG2', 'bGamma3', 'bphi', 'bphid'] + \
            ['c0', 'c2', 'c4', 'cfog'] + ['P_shot', 'a0', 'a2']

cosmo_params = {'h': 0.6736, 'omega_b': 0.02237, 'omega_c': 0.1200, 'As': 2.100e-9, 'ns': 0.9649, 'fnl':5.}
bias_params  = {'b1':2., 'b2':-1., 'bG2':0.1, 'bGamma3':-0.1, 'bphi':5., 'bphid':10.,
                'c0':5., 'c2':10., 'c4':-5., 'cfog':5., 'P_shot':1., 'a0':0., 'a2':0.}


k = np.linspace(0.01, 0.25, 25)

N = 1000
N_PROC=12

# fraction of dataset to be partitioned to the training | validation | test sets
train_frac = 0.8
valid_frac = 0.1
test_frac  = 0.1

#home_dir = "/home/u12/jadamo/CovNet/Training-Set-HighZ-NGC/"

config_dir = net_config_dir+"example.yaml"
save_dir = "/home/joeadamo/Research/Data/SPHEREx-Data/Training-Set-EFT/"

#-------------------------------------------------------------------
# FUNCTIONS
#-------------------------------------------------------------------

def prepare_ps_inputs(samples, params, config_dict):
    """takes a set of parameters and oragnizes them to the format expected by ps_1loop"""
    param_vector = []
    for pname in cosmo_list:
        if pname in params:
            #print(params.index(pname))
            param_vector.append(samples[params.index(pname)])
        else:
            param_vector.append(cosmo_params[pname])

    # The below is a (unrealistic) case in which all tracers have the same nuisance parameter values.
    for isample in range(config_dict["num_tracers"]):
        for iz in range(config_dict["num_zbins"]):
            sub_vector = []
            for pname in bias_list:
                #print(pname)
                if pname in params:
                    #print(params.index(pname))
                    sub_vector.append(samples[params.index(pname)])
                else:
                    sub_vector.append(bias_params[pname])
            #param_vector += [nuisance_params[pname] for pname in nuisance_pname_list]
            param_vector += sub_vector
        return np.array(param_vector)

def get_power_spectrum(samples, params, ps_config, config_dict):

    ells = [0, 2]
    param_vector = prepare_ps_inputs(samples, params, config_dict)

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

    # TEMP: ignore integration warnings to make output more clean
    #warnings.filterwarnings("ignore")
    
    config_dict = load_config_file(config_dir)

    params, priors = organize_parameters(config_dict)

    samples = make_latin_hypercube(priors, N)

    ps_config["redshift_list"] = config_dict["z_eff"]

    print("Generating", str(N), "power spectra...")
    print("Number of redshift bins:", len(ps_config["redshift_list"]))

    # initialize pool for multiprocessing
    t1 = time.time()
    p = Pool(processes=N_PROC)
    pk, result = zip(*p.starmap(get_power_spectrum, 
                           zip(samples, repeat(params), 
                               repeat(ps_config), repeat(config_dict))))
    p.close()
    p.join()

    # # aggregate data
    pk = np.array(pk)
    result = np.array(result)

    idx_pass = np.where(result == 0)[0]
    fail_compute = len(np.where(result == -1)[0])

    pk = pk[idx_pass]
    samples = samples[idx_pass]

    if pk.shape[0] > 1:
        np.savez(save_dir+"pk-raw.npz", params=samples, pk=pk)

        F_test = np.load(save_dir+"pk-raw.npz")
        print(F_test["params"].shape, F_test["pk"].shape)
    t2 = time.time()
    print("Done! Took {:0.0f} hours {:0.0f} minutes".format(math.floor((t2 - t1)/3600), math.floor((t2 - t1)/60%60)))
    print("Made {:0.0f} / {:0.0f} galaxy power spectra".format(N - fail_compute, N))
    print("{:0.0f} ({:0.2f}%) power spectra failed to compute".format(fail_compute, 100.*fail_compute / N))

    organize_training_set(save_dir, train_frac, valid_frac, test_frac,
                          samples.shape[1], 1, 1, len(k), True)

if __name__ == "__main__":
    main()
