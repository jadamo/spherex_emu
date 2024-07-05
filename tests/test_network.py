import torch
import numpy as np

import spherex_emu.emulator as emulator
from spherex_emu.filepaths import network_pars_dir

def test_single_sample_single_redshift_network():

    test_dir = network_pars_dir + "network_pars_single_sample_single_redshift.yaml"

    # constructes the network
    test_emulator = emulator.pk_emulator(test_dir)

    # generate a random input sequence and pass it through the network
    test_input = torch.randn(1, test_emulator.config_dict["num_cosmo_params"] + 
                                test_emulator.config_dict["num_bias_params"],
                                device = test_emulator.device)

    test_emulator.model.eval()
    test_output = test_emulator.model(test_input)

    assert torch.all(test_output >= 0) and torch.all(test_output <= 1.)
    assert test_output.shape[0] == 1
    assert test_output.shape[1] == 2 * test_emulator.config_dict["output_kbins"]

    # do the same as above except now pass it some more realistic parameters
    test_params = np.array([2.100e-9, 5., 0.6777, 0.1200, 2.0])
    test_output = test_emulator.get_power_spectra(test_params)
    assert test_output.shape == (1, 1, 2, 25)


def test_multi_sample_multi_redshift_network():

    test_dir = network_pars_dir+"network_pars_2_sample_2_redshift.yaml"

    # constructes the network
    test_emulator = emulator.pk_emulator(test_dir)

    # generate a random input sequence and pass it through the network
    test_input = torch.randn(1, test_emulator.num_cosmo_params + \
                                (test_emulator.num_bias_params *test_emulator.num_zbins * test_emulator.num_samples),
                                device = test_emulator.device)
    
    test_emulator.model.eval()
    test_output = test_emulator.model(test_input)

    assert torch.all(test_output >= 0) and torch.all(test_output <= 1.)
    assert test_output.shape[:] == (1, 6, 2*test_emulator.output_kbins)

    # do the same as above except now pass it some more realistic parameters
    test_params = np.array([2.100e-9, 5., 0.6777, 0.1200, 2.0, 1.5, 1.4, 1.3])
    test_output = test_emulator.get_power_spectra(test_params)
    assert test_output.shape == (2, 3, 2, 25)


