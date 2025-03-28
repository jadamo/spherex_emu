import torch

from spherex_emu.utils import *

def test_normalize_cosmo_params():
    test_tensor = (torch.rand(10*10) * 100) - 50
    normalizations = torch.tensor([-10, 10])
    manual_norm_tensor = (test_tensor + 10) / (20)

    norm_tensor = normalize_cosmo_params(test_tensor, normalizations)
    unnorm_tensor = norm_tensor * (normalizations[1] - normalizations[0]) + normalizations[0]

    assert torch.allclose(manual_norm_tensor, norm_tensor)
    assert torch.allclose(test_tensor, unnorm_tensor)

def test_normalize_power_spectra():
    
    test_tensor = (torch.rand((1, 3, 2, 2*25)) * 100) - 50
    test_fid_tensor = (torch.rand((3, 2, 2*25)) * 100) - 50
    test_eigvals = torch.rand(3, 2, 2*25)
    test_Q = torch.rand((3, 2, 2*25 ,2*25))

    test_Q_inv = torch.zeros_like(test_Q)
    for i in range(test_Q.shape[0]):
        for j in range(test_Q.shape[1]):
            test_Q_inv[i,j] = torch.linalg.inv(test_Q[i,j])

    norm_tensor = normalize_power_spectrum(test_tensor, test_fid_tensor, test_eigvals, test_Q)
    unnorm_tensor = un_normalize_power_spectrum(norm_tensor, test_fid_tensor, test_eigvals, test_Q, test_Q_inv)

    # numerical errors from normalization process expected to be higher than floating point percision
    assert torch.amax(test_tensor - unnorm_tensor) < 0.01

def test_get_parameter_ranges():

    input_dict = {"cosmo_params" : {"fnl" : {"value" : 0., "prior" : {"min" : -50, "max": 50}},
                                    "As" : {"value" : 2.1e-9}},
                  "nuisance_params" : {"b1_0_0" : {"value" : 1.5, "prior" : {"min" : 1, "max": 3}},
                                  "b2" : {"value" : -0.4, "prior" : {"min" : -1, "max": 0}}} }
    params, priors = get_parameter_ranges(input_dict)

    expected_params = ["fnl", "b1_0_0", "b2"]
    expected_priors = np.array([[-50, 50],
                                [1, 3],
                                [-1, 0]])
    
    assert expected_params == params
    assert np.all(np.equal(expected_priors, priors))