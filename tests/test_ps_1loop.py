import ps_theory_calculator
import numpy as np
import yaml, os

from spherex_emu.filepaths import net_config_dir
from spherex_emu.utils import *

cosmo_list = ['h', 'omega_b', 'omega_c', 'As', 'ns', 'fnl']
bias_list = ['b1', 'b2', 'bG2', 'bGamma3', 'bphi', 'bphid'] + \
            ['c0', 'c2', 'c4', 'cfog'] + ['P_shot', 'a0', 'a2']

cosmo_params = {'h': 0.6736, 'omega_b': 0.02237, 'omega_c': 0.1200, 'As': 2.100e-9, 'ns': 0.9649, 'fnl':5.}
bias_params  = {'b1_0_0':2., 'b2_0_0':-1., 'bG2_0_0':0.1, 'bGamma3_0_0':-0.1, 'bphi_0_0':5., 'bphid_0_0':10.,
                'c0_0_0':5., 'c2_0_0':10., 'c4_0_0':-5., 'cfog_0_0':5., 'P_shot_0_0':1., 'a0_0_0':0., 'a2_0_0':0.}

def test_ps_single_tracer_single_redshift():

    survey_pars_file = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    #survey_pars_file += '/configs/survey_pars_v28_base_cbe.yaml'
    survey_pars_file += '/configs/survey_pars_single_tracer_single_redshift_bin.yaml'
    with open(survey_pars_file) as file:
        survey_pars = yaml.load(file, Loader=yaml.FullLoader)
    ndens_table = np.array([[float(survey_pars['number_density_in_hinvMpc_%s' % (i+1)][j]) for j in range(survey_pars['nz'])] for i in range(survey_pars['nsample'])])

    ps_config = {}
    ps_config['number_density_table'] = ndens_table
    ps_config['redshift_list'] = [0.5] # redshift bins
    ps_config['Omega_m_ref'] = 0.3 # Omega_m value of the reference cosmology (assuming a flat LambdaCDM)

    config_dict = load_config_file(net_config_dir + "example.yaml")
    params, priors = organize_parameters(config_dict)
    
    #samples = make_latin_hypercube(priors, 1)
    samples = np.array([2.100e-9, 5., 0.6736, 0.1200, 2.])
    param_vector = prepare_ps_inputs(samples, params, 1, 1, cosmo_params, bias_params)
    k = np.linspace(0.01, 0.25, 25)
    print(param_vector)
    ells = [0, 2]

    theory = ps_theory_calculator.PowerSpectrumMultipole1Loop(ps_config)
    galaxy_ps = theory(k, ells, param_vector)
    params_dict = {**cosmo_params, **bias_params}

    # test parameter order was correctly passed over
    for key in theory.params:
        if key in params_dict:
            assert params_dict[key] == theory.params[key]
    