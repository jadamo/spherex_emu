# import ps_theory_calculator
import numpy as np
import yaml, os
import pytest

from mentat_lss.emulator import ps_emulator
import mentat_lss._vendor.symbolic_pofk.linear as linear
from mentat_lss.models.analytic_terms import analytic_eft_model

def test_symbolic_pofk():
    k = np.linspace(0.01, 0.2, 25)
    test_plin = linear.plin_emulated(k, 0.8, 0.25, 0.05, 0.67, 0.96859)
    assert np.all(np.isinf(test_plin)) == False
    assert np.all(np.isnan(test_plin)) == False


# @pytest.mark.parametrize("P_shot, num_tracers, num_zbins, kbins, expected", [
#     (0., 1, 1, np.linspace(0.01, 0.2, 25), np.zeros(25))
# ])
# def test_shotnoise_term(P_shot, num_tracers, num_zbins, kbins, expected):
#     redshift_list = [0]*num_zbins
#     ndens = np.random.rand(num_zbins, num_tracers)

#     model = analytic_eft_model(num_tracers, redshift_list, [0,2], kbins, ndens)
#     emu_params = []

# def test_ps_multi_sample_multi_redshift():

#     config_dir = os.path.dirname(os.path.realpath(__file__)) + "/../configs/"
#     cosmo_dict = load_config_file(config_dir+"cosmo_pars/cosmo_pars_example.yaml")
#     survey_pars = load_config_file(config_dir + 'survey_pars/survey_pars_2_tracer_2_redshift.yaml')
    
#     ndens_table = np.array([[float(survey_pars['number_density_in_hinvMpc_%s' % (i+1)][j]) for j in range(survey_pars['nz'])] for i in range(survey_pars['nsample'])])
#     z_eff = (np.array(survey_pars["zbin_lo"]) + np.array(survey_pars["zbin_hi"])) / 2.
    
#     # table of number densities of tracer samples. 
#     # (i, j) component is the number density of the i-th sample at the j-th redshift.
#     ps_config = {}
#     ps_config['number_density_table'] = ndens_table
#     ps_config['redshift_list'] = list(z_eff) # redshift bins
#     ps_config['Omega_m_ref'] = 0.3 # Omega_m value of the reference cosmology (assuming a flat LambdaCDM)
    
#     # set some parameters to be different from their fiducial values
#     sample = {"As": 2.2e-9, "h" : 0.7, "galaxy_bias_10_0_0" : 1.5, "galaxy_bias_10_0_1" : 1.6, "galaxy_bias_G2" : 0.2}

#     param_vector = prepare_ps_inputs(sample, cosmo_dict, 2, len(z_eff))
#     k = np.linspace(0.01, 0.25, 25)
#     ells = [0, 2]

#     theory = ps_theory_calculator.PowerSpectrumMultipole1Loop(ps_config)
#     galaxy_ps = theory(k, ells, param_vector)
#     assert galaxy_ps.shape == (len(z_eff), 3, 2, 25)

#     # test parameter order was correctly passed over
#     for key in theory.params:
#         if "omega_" not in key: 
#             key1 = key.split("_", 1)[0]
#             key2 = "_".join(key.split("_", 2)[:2])
#         else: key1, key2 = key, key

#         if key in list(sample.keys()):
#             assert theory.params[key] == sample[key]
#         elif key1 in list(sample.keys()):
#             assert theory.params[key] == sample[key1]
#         elif key2 in list(sample.keys()):
#             assert theory.params[key] == sample[key2]
#         elif key in list(cosmo_dict.keys()):
#             assert theory.params[key] == cosmo_dict[key]
#         elif key1 in list(cosmo_dict.keys()):
#             assert theory.params[key] == cosmo_dict[key1]