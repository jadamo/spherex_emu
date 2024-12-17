import numpy as np

import ps_theory_calculator
from spherex_emu.utils import prepare_ps_inputs, load_config_file
import spherex_emu.filepaths as filepaths 

def main():
    cosmo_dict = load_config_file(filepaths.cosmo_pars_dir+"eft_single_sample_single_redshift.yaml")
    survey_pars = load_config_file(filepaths.survey_pars_dir + 'survey_pars_single_sample_single_redshift.yaml')
    
    ndens_table = np.array([[float(survey_pars['number_density_in_hinvMpc_%s' % (i+1)][j]) for j in range(survey_pars['nz'])] for i in range(survey_pars['nsample'])])
    z_eff = (np.array(survey_pars["zbin_lo"]) + np.array(survey_pars["zbin_hi"])) / 2.
    
    # table of number densities of tracer samples. 
    # (i, j) component is the number density of the i-th sample at the j-th redshift.
    ps_config = {}
    ps_config['number_density_table'] = ndens_table
    ps_config['redshift_list'] = list(z_eff) # redshift bins
    ps_config['Omega_m_ref'] = 0.3 # Omega_m value of the reference cosmology (assuming a flat LambdaCDM)
    
    # set parameters to be different from their fiducial values here
    #alternate_params = {"As": 2.2e-9, "h" : 0.7, "b1_0_0" : 1.5, "b1_0_1" : 1.6, "bG2" : 0.2}
    alternate_params = {}

    param_vector = prepare_ps_inputs(alternate_params, cosmo_dict, ndens_table.shape[0], len(z_eff))
    #k = np.linspace(0.01, 0.25, 25)
    k_data = np.load('/home/joeadamo/Research/SPHEREx/covapt_mt/data/input_data/k_emu_test.npz')
    k = k_data["k_0"]
    # k = np.array([0.00694, 0.01482, 0.0227, 0.03058, 0.03846, 0.04634, 0.05422, 0.0621, 0.06998,
    #           0.07786, 0.08574, 0.09362, 0.1015,  0.10938, 0.11726, 0.12514, 0.13302, 0.1409,
    #           0.14878, 0.15666, 0.16454, 0.17242, 0.1803,  0.18818, 0.19606])
    # k = np.array([0.002093, 0.004879 ,0.007665 ,0.010451, 0.013237 ,0.016023 ,0.018809 ,0.021595,0.024381, 0.027167, 0.029953, 0.032739, 0.035525, 0.038311, 0.041097, 0.043883,
    #      0.046669, 0.049455, 0.052241, 0.055027, 0.057813, 0.060599, 0.063385, 0.066171,
    #      0.068957, 0.071743, 0.074529, 0.077315, 0.080101, 0.082887, 0.085673, 0.088459,
    #      0.091245, 0.094031, 0.096817, 0.099603, 0.102389, 0.105175, 0.107961, 0.110747,
    #      0.113533, 0.116319, 0.119105, 0.121891, 0.124677, 0.127463, 0.130249, 0.133035,
    #      0.135821, 0.138607]) / 0.7
    ells = [0, 2]

    print(param_vector)
    theory = ps_theory_calculator.PowerSpectrumMultipole1Loop(ps_config)
    galaxy_ps = theory(k, ells, param_vector) / cosmo_dict["cosmo_params"]["h"]["value"]**3

    print(galaxy_ps.shape)
    np.save('/home/joeadamo/Research/SPHEREx/covapt_mt/data/input_data/ps_emu_test_1_tracer.npy', galaxy_ps)

if __name__ == "__main__":
    main()
