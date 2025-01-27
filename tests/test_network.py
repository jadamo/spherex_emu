import torch
import numpy as np

import spherex_emu.emulator as emulator
from spherex_emu.filepaths import network_pars_dir
from spherex_emu.models.blocks import linear_with_channels

# def test_single_sample_single_redshift_network():

#     test_dir = network_pars_dir + "network_pars_single_sample_single_redshift.yaml"

#     # constructes the network
#     test_emulator = emulator.pk_emulator(test_dir)

#     # generate a random input sequence and pass it through the network
#     test_input = torch.randn(1, test_emulator.config_dict["num_cosmo_params"] + 
#                                 test_emulator.config_dict["num_bias_params"],
#                                 device = test_emulator.device)

#     test_emulator.model.eval()
#     test_output = test_emulator.model(test_input)

#     #assert torch.all(test_output >= 0) and torch.all(test_output <= 1.)
#     assert test_output.shape[0] == 1
#     assert test_output.shape[1] == 2 * test_emulator.num_kbins

#     # do the same as above except now pass it some more realistic parameters
#     test_params = np.array([2.100e-9, 5., 0.6777, 0.1200, 2.0])
#     test_output = test_emulator.get_power_spectra(test_params)
#     assert test_output.shape == (1, 1, 2, 25)

def test_linear_with_channels():
    # test that the linear_with_channels sub-block treats channels independently

    test_input = torch.rand((1, 2, 10))
    parallel_layers = linear_with_channels(10, 10, 2)
    parallel_layers.initialize_params("He")
    with torch.no_grad():
        parallel_layers.w[0] = 1.
        parallel_layers.b[0] = 0.
    test_output = parallel_layers(test_input)

    assert torch.all(test_output[:,0] == torch.sum(test_input[:,0]))
    assert torch.all(test_output[:,1] != torch.sum(test_input[:,1]))

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

    #assert torch.all(test_output >= 0) and torch.all(test_output <= 1.)
    assert test_output.shape == (1, test_emulator.num_zbins, test_emulator.num_spectra*
                                                             test_emulator.num_kbins*
                                                             test_emulator.num_ells)

    # do the same as above except now pass it some more realistic parameters
    # test_params = np.array([2.100e-9, 5., 0.6777, 0.1200, 2.0, 1.5, 1.4, 1.3])
    # test_output = test_emulator.get_power_spectra(test_params)
    # assert test_output.shape == (2, 3, 2, 25)


