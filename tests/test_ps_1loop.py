import ps_theory_calculator
import numpy as np
import yaml, os

import spherex_emu.filepaths as filepaths
from spherex_emu.utils import *

def test_ps_single_tracer_single_redshift():

    cosmo_pars_file = filepaths.cosmo_pars_dir+"eft_single_tracer_fiducial.yaml"
    with open(cosmo_pars_file) as file:
        fiducial_cosmology = yaml.load(file, Loader=yaml.FullLoader)
    survey_pars_file = filepaths.survey_pars_dir + 'survey_pars_single_tracer_single_redshift.yaml'
    with open(survey_pars_file) as file:
        survey_pars = yaml.load(file, Loader=yaml.FullLoader)
    
    ndens_table = np.array([[float(survey_pars['number_density_in_hinvMpc_%s' % (i+1)][j]) for j in range(survey_pars['nz'])] for i in range(survey_pars['nsample'])])

    ps_config = {}
    ps_config['number_density_table'] = ndens_table
    ps_config['redshift_list'] = [0.5] # redshift bins
    ps_config['Omega_m_ref'] = 0.3 # Omega_m value of the reference cosmology (assuming a flat LambdaCDM)

    config_dict = load_config_file(filepaths.network_pars_dir + "example.yaml")
    params_names, param_ranges = get_parameter_ranges(config_dict)
    
    #samples = make_latin_hypercube(priors, 1)
    sample = {"As": 2.2e-9, "h" : 0.7, "b1" : 1.5}
    param_vector = prepare_ps_inputs(sample, fiducial_cosmology, 1, 1)
    k = np.linspace(0.01, 0.25, 25)
    ells = [0, 2]

    theory = ps_theory_calculator.PowerSpectrumMultipole1Loop(ps_config)
    galaxy_ps = theory(k, ells, param_vector)

    # test parameter order was correctly passed over
    for key in theory.params:
        if key in list(sample.keys()):
            assert theory.params[key] == sample[key]
        elif key in list(fiducial_cosmology.keys()):
            assert theory.params[key] == fiducial_cosmology[key]
    