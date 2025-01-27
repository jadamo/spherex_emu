import torch

from spherex_emu.utils import *
from spherex_emu.filepaths import cosmo_pars_dir

def test_normalize_cosmo_params():
    test_tensor = (torch.rand(10*10) * 100) - 50
    manual_norm_tensor = (test_tensor + 10) / (20)

    norm_tensor = normalize_cosmo_params(test_tensor, torch.tensor([-10, 10]))
    unnorm_tensor = un_normalize(norm_tensor, torch.tensor([-10, 10]))

    assert torch.allclose(manual_norm_tensor, norm_tensor)
    assert torch.allclose(test_tensor, unnorm_tensor)

def test_normalize_power_spectra():
    
    test_tensor = (torch.rand((1, 2, 3*2*25)) * 100) - 50
    test_fid_tensor = (torch.rand((2, 3*2*25)) * 100) - 50
    test_inv_cov = (torch.eye(3*2*25))
    test_inv_cov = test_inv_cov.repeat(2, 1, 1)

    norm_tensor = normalize_power_spectrum_diagonal(test_tensor, test_fid_tensor, test_inv_cov)
    unnorm_tensor = un_normalize_power_spectrum_diagonal(norm_tensor, test_fid_tensor, test_inv_cov)

    assert torch.allclose(test_tensor, unnorm_tensor)

def test_get_parameter_ranges():

    input_dict = {"cosmo_params" : {"fnl" : {"value" : 0., "prior" : {"min" : -50, "max": 50}},
                                    "As" : {"value" : 2.1e-9}},
                  "bias_params" : {"b1_0_0" : {"value" : 1.5, "prior" : {"min" : 1, "max": 3}},
                                  "b2" : {"value" : -0.4, "prior" : {"min" : -1, "max": 0}}} }
    params, priors = get_parameter_ranges(input_dict)

    expected_params = ["fnl", "b1_0_0", "b2"]
    expected_priors = np.array([[-50, 50],
                                [1, 3],
                                [-1, 0]])
    
    assert expected_params == params
    assert np.all(np.equal(expected_priors, priors))