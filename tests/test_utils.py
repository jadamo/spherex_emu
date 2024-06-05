import torch

from spherex_emu.utils import *
from spherex_emu.filepaths import network_pars_dir

# def test_symmetric_log():

#     # first test that this works for all positive values
#     test_tensor_all_positive = torch.rand(10, 10) * 100

#     log_tensor = symmetric_log(test_tensor_all_positive)
#     unlog_tensor = symmetric_exp(log_tensor)

#     assert torch.allclose(test_tensor_all_positive, unlog_tensor)

#     # now test that it works for a mix of positive and negative values
#     test_tensor = (torch.rand(5, 50) * 100) - 50

#     log_tensor = symmetric_log(test_tensor)
#     unlog_tensor = symmetric_exp(log_tensor)

#     assert torch.allclose(test_tensor, unlog_tensor)

def test_normalize():
    test_tensor = (torch.rand(10*10) * 100) - 50
    manual_norm_tensor = (test_tensor + 10) / (20)

    norm_tensor = normalize(test_tensor, torch.tensor([-10, 10]))
    unnorm_tensor = un_normalize(norm_tensor, torch.tensor([-10, 10]))

    assert torch.allclose(manual_norm_tensor, norm_tensor)
    assert torch.allclose(test_tensor, unnorm_tensor)

def test_get_parameter_ranges():

    test_dir = network_pars_dir + "example.yaml"

    config_dict = load_config_file(test_dir)

    params, priors = get_parameter_ranges(config_dict)

    expected_params = ["As", "fnl", "h", "omega_c", "b1_0"]
    expected_priors = np.array([[1.2e-9, 2.7e-9],
                                [-50, 50],
                                [0.4, 1.0],
                                [0.05, 0.3],
                                [1., 4.]])
    
    assert expected_params == params
    assert np.all(np.equal(expected_priors, priors))