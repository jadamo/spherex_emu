import ps_theory_calculator
import numpy as np
import yaml, os

import spherex_emu.filepaths as filepaths
from spherex_emu.utils import *

def test_ps_single_sample_single_redshift():

    fiducial_cosmology = load_config_file(filepaths.cosmo_pars_dir+"eft_single_sample_fiducial.yaml")
    survey_pars = load_config_file(filepaths.survey_pars_dir + 'survey_pars_single_sample_single_redshift.yaml')
    
    ndens_table = np.array([[float(survey_pars['number_density_in_hinvMpc_%s' % (i+1)][j]) for j in range(survey_pars['nz'])] for i in range(survey_pars['nsample'])])

    ps_config = {}
    ps_config['number_density_table'] = ndens_table
    ps_config['redshift_list'] = [0.5] # redshift bins
    ps_config['Omega_m_ref'] = 0.3 # Omega_m value of the reference cosmology (assuming a flat LambdaCDM)
    
    #samples = make_latin_hypercube(priors, 1)
    sample = {"As": 2.2e-9, "h" : 0.7, "b1" : 1.5}
    param_vector = prepare_ps_inputs(sample, fiducial_cosmology, 1, 1)
    k = np.linspace(0.01, 0.25, 25)
    ells = [0, 2]

    theory = ps_theory_calculator.PowerSpectrumMultipole1Loop(ps_config)
    galaxy_ps = theory(k, ells, param_vector)
    assert galaxy_ps.shape == (1, 1, 2, 25)

    # test parameter order was correctly passed over
    for key in theory.params:
        if key in list(sample.keys()):
            assert theory.params[key] == sample[key]
        elif key in list(fiducial_cosmology.keys()):
            assert theory.params[key] == fiducial_cosmology[key]
    

def test_ps_single_sample_multi_redshift():

    fiducial_cosmology = load_config_file(filepaths.cosmo_pars_dir+"eft_single_sample_fiducial.yaml")
    survey_pars = load_config_file(filepaths.survey_pars_dir + 'survey_pars_single_sample_2_redshift.yaml')
    
    ndens_table = np.array([[float(survey_pars['number_density_in_hinvMpc_%s' % (i+1)][j]) for j in range(survey_pars['nz'])] for i in range(survey_pars['nsample'])])
    z_eff = (np.array(survey_pars["zbin_lo"]) + np.array(survey_pars["zbin_hi"])) / 2.
    
    # table of number densities of tracer samples. 
    # (i, j) component is the number density of the i-th sample at the j-th redshift.
    ps_config = {}
    ps_config['number_density_table'] = ndens_table
    ps_config['redshift_list'] = list(z_eff) # redshift bins
    ps_config['Omega_m_ref'] = 0.3 # Omega_m value of the reference cosmology (assuming a flat LambdaCDM)
    
    #samples = make_latin_hypercube(priors, 1)
    sample = {"As": 2.2e-9, "h" : 0.7, "b1" : 1.5}
    param_vector = prepare_ps_inputs(sample, fiducial_cosmology, 1, len(z_eff))
    k = np.linspace(0.01, 0.25, 25)
    ells = [0, 2]

    theory = ps_theory_calculator.PowerSpectrumMultipole1Loop(ps_config)
    galaxy_ps = theory(k, ells, param_vector)
    assert galaxy_ps.shape == (len(z_eff), 1, 2, 25)

    # test parameter order was correctly passed over
    for key in theory.params:
        if "omega_" not in key: 
            key1 = key.split("_", 1)[0]
        else: key1 = key

        if key in list(sample.keys()):
            assert theory.params[key] == sample[key]
        elif key1 in list(sample.keys()):
            assert theory.params[key] == sample[key1]
        elif key in list(fiducial_cosmology.keys()):
            assert theory.params[key] == fiducial_cosmology[key]
        elif key1 in list(fiducial_cosmology.keys()):
            assert theory.params[key] == fiducial_cosmology[key1]

def test_ps_multi_sample_multi_redshift():

    fiducial_cosmology = load_config_file(filepaths.cosmo_pars_dir+"eft_single_sample_fiducial.yaml")
    survey_pars = load_config_file(filepaths.survey_pars_dir + 'survey_pars_2_sample_2_redshift.yaml')
    
    ndens_table = np.array([[float(survey_pars['number_density_in_hinvMpc_%s' % (i+1)][j]) for j in range(survey_pars['nz'])] for i in range(survey_pars['nsample'])])
    z_eff = (np.array(survey_pars["zbin_lo"]) + np.array(survey_pars["zbin_hi"])) / 2.
    
    # table of number densities of tracer samples. 
    # (i, j) component is the number density of the i-th sample at the j-th redshift.
    ps_config = {}
    ps_config['number_density_table'] = ndens_table
    ps_config['redshift_list'] = list(z_eff) # redshift bins
    ps_config['Omega_m_ref'] = 0.3 # Omega_m value of the reference cosmology (assuming a flat LambdaCDM)
    
    # set some parameters to be different from their fiducial values
    sample = {"As": 2.2e-9, "h" : 0.7, "b1_0" : 1.5, "b1_1" : 1.6}

    param_vector = prepare_ps_inputs(sample, fiducial_cosmology, 2, len(z_eff))
    k = np.linspace(0.01, 0.25, 25)
    ells = [0, 2]

    theory = ps_theory_calculator.PowerSpectrumMultipole1Loop(ps_config)
    galaxy_ps = theory(k, ells, param_vector)
    assert galaxy_ps.shape == (len(z_eff), 3, 2, 25)

    # test parameter order was correctly passed over
    for key in theory.params:
        if "omega_" not in key: 
            key1 = key.split("_", 1)[0]
            key2 = "_".join(key.split("_", 2)[:2])
        else: key1, key2 = key, key

        if key in list(sample.keys()):
            assert theory.params[key] == sample[key]
        elif key1 in list(sample.keys()):
            assert theory.params[key] == sample[key1]
        elif key2 in list(sample.keys()):
            assert theory.params[key] == sample[key2]
        elif key in list(fiducial_cosmology.keys()):
            assert theory.params[key] == fiducial_cosmology[key]
        elif key1 in list(fiducial_cosmology.keys()):
            assert theory.params[key] == fiducial_cosmology[key1]