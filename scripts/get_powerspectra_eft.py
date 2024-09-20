import time, math, yaml
import numpy as np

import ps_theory_calculator
from spherex_emu.utils import prepare_ps_inputs, load_config_file
import spherex_emu.filepaths as filepaths 

def main():
    cosmo_dict = load_config_file(filepaths.cosmo_pars_dir+"eft_2_sample_2_redshift.yaml")
    survey_pars = load_config_file(filepaths.survey_pars_dir + 'survey_pars_2_sample_2_redshift.yaml')
    
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

    param_vector = prepare_ps_inputs(alternate_params, cosmo_dict, 2, len(z_eff))
    k = np.linspace(0.01, 0.25, 25)
    ells = [0, 2, 4]

    theory = ps_theory_calculator.PowerSpectrumMultipole1Loop(ps_config)
    galaxy_ps = theory(k, ells, param_vector)

    print(galaxy_ps.shape)
    np.save(filepaths.data_dir+"ps_eft_fid.npy", galaxy_ps)

if __name__ == "__main__":
    main()