import torch
import numpy as np
import os

import spherex_emu.emulator as emulator
from spherex_emu.models.blocks import *

def test_linear_with_channels():
    # test that the linear_with_channels sub-block treats channels independently

    test_input = torch.rand((1, 2, 10))
    parallel_layers = linear_with_channels(10, 10, 2)
    with torch.no_grad():
        parallel_layers.w[0] = 1.
        parallel_layers.b[0] = 0.
        parallel_layers.w[1] = 0.
        parallel_layers.b[1] = 0.
    test_output = parallel_layers(test_input)

    assert torch.all(test_output[:,0] == torch.sum(test_input[:,0]))
    assert torch.all(test_output[:,1] != torch.sum(test_input[:,1]))

def test_block_resnet():

    test_input = torch.rand((100, 10))
    resnet_block = block_resnet(10, 10, 2, True)
    test_output = resnet_block(test_input)

    assert test_output.shape == (100, 10)
    assert not torch.all(torch.isnan(test_output))
    assert not torch.all(torch.isinf(test_output))

def test_stacked_transformer_network():

    current_dir = os.path.dirname(os.path.realpath(__file__))
    test_dir = current_dir+"/../configs/network_pars/network_pars_example.yaml"

    # constructes the network
    test_emulator = emulator.pk_emulator(test_dir, "train")

    # generate a random input sequence and pass it through the network
    test_input = torch.randn(1, test_emulator.num_cosmo_params + \
                                (test_emulator.num_nuisance_params *test_emulator.num_zbins * test_emulator.num_tracers),
                                device = test_emulator.device)
    test_emulator.galaxy_ps_model.eval()
    test_input = test_emulator.galaxy_ps_model.organize_parameters(test_input)

    test_output_sub = test_emulator.galaxy_ps_model.forward(test_input, 0)
    test_output_full = test_emulator.galaxy_ps_model.forward(test_input)

    assert torch.all(torch.isnan(test_output_full)) == False
    assert torch.all(torch.isinf(test_output_full)) == False

    assert test_output_sub.shape == (1, test_emulator.num_kbins * test_emulator.num_ells)
    assert test_output_full.shape == (1, test_emulator.num_spectra,
                                         test_emulator.num_zbins,
                                         test_emulator.num_kbins*test_emulator.num_ells)
    assert torch.allclose(test_output_sub, test_output_full[:,0,0])


