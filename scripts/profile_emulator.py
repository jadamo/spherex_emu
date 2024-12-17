from spherex_emu.emulator import pk_emulator
from spherex_emu.filepaths import network_pars_dir
import math, time
import torch

import ps_theory_calculator
from spherex_emu.utils import *
import spherex_emu.filepaths as filepaths 

net_config_file = filepaths.network_pars_dir+"network_pars_single_sample_single_redshift.yaml"
cosmo_config_file = filepaths.cosmo_pars_dir+"eft_single_sample_single_redshift.yaml"
survey_config_file = filepaths.survey_pars_dir+'survey_pars_single_sample_single_redshift.yaml'

k_min = 0.01
k_max = 0.2

def profile_eft_power_spectrum(param_names, param_ranges, cosmo_dict, ps_config, k, N):

    params = make_latin_hypercube(param_ranges, N)
    ells = ps_config["ells"]
    num_samples = ps_config['number_density_table'].shape[0]
    num_zbins = len(ps_config["redshift_list"])

    t_avg = 0
    for i in range(N):
        sample_dict = dict(zip(param_names, params[i]))
        param_vector = prepare_ps_inputs(sample_dict, cosmo_dict, num_samples, num_zbins)
        t_start = time.time()
        theory = ps_theory_calculator.PowerSpectrumMultipole1Loop(ps_config)
        galaxy_ps = theory(k, ells, param_vector) / cosmo_dict["cosmo_params"]["h"]["value"]**3
        t_avg += time.time() - t_start

    return t_avg / N

def profile_emu_power_spectrum(net, N, params_size):

    t_avg = 0
    for i in range(N):
        params = np.random.rand(params_size)
        t_start = time.time()
        test = net.get_power_spectra(params)
        t_avg += time.time() - t_start
    return t_avg / N

def main():

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
    ps_config["ells"] = [0,1,2,3,4]

    param_names, param_ranges = get_parameter_ranges(cosmo_dict)
    k_bins = 50

    z_array = [0.1, 0.3, 0.5, 0.7, 0.9, 1.3, 1.9, 2.5, 3.1, 3.7, 4.3] 

    t_data = np.zeros((3, 5, 3))

    for i_tracer in range(1, 5+1):
        for nz in range(1, 3+1):
            ndens_table = np.array([[0.000501 for j in range(nz)] for i in range(i_tracer)])
            z_eff = z_array[:nz]
            num_spectra = ndens_table.shape[0] + math.comb(ndens_table.shape[0], 2)

            # first run the eft code
            ps_config = {}
            ps_config['number_density_table'] = ndens_table
            ps_config['redshift_list'] = z_eff # redshift bins
            ps_config['Omega_m_ref'] = 0.3 # Omega_m value of the reference cosmology (assuming a flat LambdaCDM)
            ps_config["ells"] = [0,1,2,3,4]
            k = np.linspace(k_min, k_max, k_bins)
            N = 6
            t_eft = profile_eft_power_spectrum(param_names, param_ranges, cosmo_dict, ps_config, k, N)
            #t_eft = 0

            # next, run the emulator code
            config_dict["num_cosmo_params"] = 2
            config_dict["cosmo_dir"] = ""
            config_dict["num_zbins"] = nz
            config_dict["num_samples"] = i_tracer

            output_size = k_bins*len(ps_config["ells"]*num_spectra*nz)
            config_dict["mlp_dims"] = [output_size, output_size, output_size]
            config_dict["split_dim"] = 20 * num_spectra
            config_dict["split_size"] = 20
            emulator = pk_emulator(config_dict=config_dict)
            emulator.model.eval()
            total_params = sum(p.numel() for p in emulator.model.parameters() if p.requires_grad)

            t_emu = profile_emu_power_spectrum(emulator, 100, 2 + 2*nz*i_tracer)

            print("avg time for eft ({:0.0f} tracers, {:0.0f} z bins) = {:0.2f}s".format(i_tracer, nz, t_eft))
            print("avg time for emulator = {:0.4f}s, factor speedup = {:0.2f}x".format(t_emu, t_eft / t_emu))

            t_data[0,i_tracer-1, nz-1] = t_eft
            t_data[1,i_tracer-1, nz-1] = t_emu
            t_data[2,i_tracer-1, nz-1] = total_params
            np.save("../data/profile_data.npy", t_data)

if __name__ == "__main__":
    main()