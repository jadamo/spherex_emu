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

    norm_tensor = normalize_power_spectrum(test_tensor, test_fid_tensor, test_inv_cov)
    unnorm_tensor = un_normalize_power_spectrum(norm_tensor, test_fid_tensor, test_inv_cov)

    assert torch.allclose(test_tensor, unnorm_tensor)

def test_get_parameter_ranges():

    test_dir = cosmo_pars_dir + "eft_single_sample_single_redshift.yaml"

    config_dict = load_config_file(test_dir)
    params, priors = get_parameter_ranges(config_dict)

    expected_params = ["h", "omega_c", "As", "fnl", "b1"]
    expected_priors = np.array([[0.4, 1.0],
                                [0.05, 0.3],
                                [1.2e-9, 2.7e-9],
                                [-50, 50],
                                [1., 4.]])
    
    assert expected_params == params
    assert np.all(np.equal(expected_priors, priors))