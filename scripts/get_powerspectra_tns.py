import numpy as np
import camb, itertools
from camb import model
from math import comb

#from mentat_lss.fastpt import FASTPT
from mentat_lss.gpsclass import CalcGalaxyPowerSpec
from mentat_lss.utils import prepare_ps_inputs, load_config_file, fgrowth

#Creates galaxy power spectra from a set of input parameters - input into galaxy ps class
def get_power_spectrum(params, k, num_tracers, z_eff):

    params[0] = params[0] * 100
    num_spectra = num_tracers + comb(num_tracers, 2)
    
    num_cosmo_params = 5
    num_bias_params = 5
    num_nuisance_params = 8 # <- HACK: includes stochastic parameters
    pk = np.zeros((len(z_eff), num_spectra, 2, len(k)))

    nonlin_model = CalcGalaxyPowerSpec(z_eff,k,params[:num_cosmo_params])

    for z_idx in range(len(z_eff)):
        
        sample_idx = 0
        for isample1, isample2 in itertools.product(range(num_tracers), repeat=2):
            if isample1 > isample2: continue

            # retrieve relavent bias parameters
            idx1 = num_cosmo_params + (isample1*num_nuisance_params) + (z_idx*num_tracers*num_nuisance_params)
            idx2 = num_cosmo_params + (isample2*num_nuisance_params) + (z_idx * num_nuisance_params*num_tracers)
            bias1 = params[idx1:idx1+num_bias_params]
            bias2 = params[idx2:idx2+num_bias_params]

            pk[z_idx, sample_idx, 0, :] = nonlin_model.get_nonlinear_ps(z_idx, 0, bias1, bias2)
            pk[z_idx, sample_idx, 1, :] = nonlin_model.get_nonlinear_ps(z_idx, 2, bias1, bias2)
            sample_idx+=1
    return pk

def main():
    
    input_dir = "/home/joeadamo/Research/SPHEREx/mentat_lss/"
    cosmo_dict = load_config_file(input_dir+"configs/cosmo_pars/cosmo_pars_tns.yaml")
    survey_pars = load_config_file(input_dir+"configs/survey_pars/survey_pars_2_tracer_2_redshift.yaml")
    k_array_file = "/home/joeadamo/Research/Data/SPHEREx-Data/training_set_eft_high_boss_cut/kbins.npz"
    save_file = "/home/joeadamo/Research/Data/SPHEREx-Data/ps_tns.npy"

    ndens_table = np.array([[float(survey_pars['number_density_in_hinvMpc_%s' % (i+1)][j]) for j in range(survey_pars['nz'])] for i in range(survey_pars['nsample'])])
    num_tracers = ndens_table.shape[0]
    z_eff = (np.array(survey_pars["zbin_lo"]) + np.array(survey_pars["zbin_hi"])) / 2.

    # TODO: Place this info in a better location like a survey pars file
    k_data = np.load(k_array_file)
    k = k_data["k"]
    alternate_params = {}

    param_vector = prepare_ps_inputs(alternate_params, cosmo_dict, 2, len(z_eff))
    k, pk = get_power_spectrum(param_vector, k, num_tracers, z_eff)
    
    print("saving to", save_file)
    np.save(save_file, pk)

if __name__ == "__main__":
    main()