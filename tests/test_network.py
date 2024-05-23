import pytest, os
import torch

import spherex_emu.emulator as emulator

def test_single_tracer_network():

    test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    test_dir+="/configs/example.yaml"

    # constructes the network
    test_emulator = emulator.pk_emulator(test_dir)

    # generate a random input sequence and pass it through the network
    test_input = torch.randn(1, test_emulator.config_dict["num_cosmo_params"] + 
                                test_emulator.config_dict["num_bias_params"])

    test_emulator.model.eval()
    test_output = test_emulator.model(test_input)

    assert test_output.shape[0] == 1
    assert test_output.shape[1] == 2 * test_emulator.config_dict["output_kbins"]