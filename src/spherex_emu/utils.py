import yaml, os
from scipy.stats import qmc
from scipy.special import hyp2f1
from torch.nn import functional as F
import numpy as np
import itertools
import torch


def load_config_file(config_file:str):
    """loads in the emulator config file as a dictionary object
    
    Args:
        config_file: Config file path and name to laod
    """
    with open(config_file, "r") as file:
        try:
            config_dict = yaml.load(file, Loader=yaml.FullLoader)
        except:
            print("ERROR! Couldn't read yaml file")
            return None
    
    return config_dict


def get_parameter_ranges(cosmo_dict):
    """Returns cosmology and bias parameter priors based on the input cosmo_dict
    TODO: Upgrade to handle normal priors as well as uniform priors"""

    cosmo_params = {}
    for param in cosmo_dict["cosmo_params"]:
        if "prior" in cosmo_dict["cosmo_params"][param]:
            cosmo_params[param] = [cosmo_dict["cosmo_params"][param]["prior"]["min"],
                                   cosmo_dict["cosmo_params"][param]["prior"]["max"]]

    nuisance_params = {}
    for param in cosmo_dict["nuisance_params"]:
        if "prior" in cosmo_dict["nuisance_params"][param]:
            nuisance_params[param] = [cosmo_dict["nuisance_params"][param]["prior"]["min"],
                                      cosmo_dict["nuisance_params"][param]["prior"]["max"]]

    params_dict = {**cosmo_params, **nuisance_params}
    priors = np.array(list(params_dict.values()))
    params = list(params_dict.keys())
    return params, priors


def prepare_ps_inputs(sample, cosmo_dict, num_tracers, num_zbins):
    """takes a set of parameters and oragnizes them to the format expected by ps_1loop"""
    param_vector = []
    # fill in cosmo params in the order ps_1loop expects
    for pname in list(cosmo_dict["cosmo_param_names"]):
        if pname in sample:
            #print(params.index(pname))
            param_vector.append(sample[pname])
        else:
            param_vector.append(cosmo_dict["cosmo_params"][pname]["value"])

    # fill in bias params
    for isample in range(num_tracers):
        for iz in range(num_zbins):
            sub_vector = []
            for pname in list(cosmo_dict["bias_param_names"] + 
                              cosmo_dict["counterterm_param_names"] + 
                              cosmo_dict["stochastic_param_names"]):
                key = pname+"_"+str(isample)+"_"+str(iz)

                if key in sample:
                    sub_vector.append(sample[key])
                elif pname in sample:
                    sub_vector.append(sample[pname])
                elif key in cosmo_dict["nuisance_params"]:
                    sub_vector.append(cosmo_dict["nuisance_params"][key]["value"])
                # special cases when bias parameters depend on other bias parameter values (TNS model)
                elif pname == "bs2" and cosmo_dict["nuisance_params"][pname]["value"] == -99:
                    sub_vector.append((-4./7)*(cosmo_dict["nuisance_params"]["b1"+"_"+str(isample)+"_"+str(iz)]["value"]-1))
                elif pname == "b3nl" and cosmo_dict["nuisance_params"][pname]["value"] == -99:
                    sub_vector.append((32./315)*(cosmo_dict["nuisance_params"]["b1"+"_"+str(isample)+"_"+str(iz)]["value"]-1))
                else:
                    sub_vector.append(cosmo_dict["nuisance_params"][pname]["value"])
            param_vector += sub_vector

    return np.array(param_vector)


def make_latin_hypercube(priors, N):
    """Generates a latin hypercube of N samples with lower and upper bounds given by priors"""

    n_dim = priors.shape[0]

    sampler = qmc.LatinHypercube(d=n_dim)
    params = sampler.random(n=N)

    for i in range(params.shape[1]):
        params[:,i] = (params[:,i] * (priors[i, 1] - priors[i, 0])) + priors[i,0]
    
    return params

def make_hypersphere(priors, dim, N):
    """Generates a hypersphere of N samples using the method from https://arxiv.org/abs/2405.01396v1"""

    # generate points in a uniform hypersphere with radius 1
    sphere_points = np.random.multivariate_normal(np.zeros(dim), np.eye(dim), size=N)
    radius = np.sqrt(np.sum(sphere_points**2, axis=1))[:, np.newaxis]
    uniform_points = np.random.uniform(0, 1, size=(N, 1))
    new_radius = uniform_points**(1./dim)
    sphere_points = (sphere_points / radius) * new_radius

    # expand each dimension to match the prior boundaries
    for d in range(dim):
        sphere_points[:,d] = ((sphere_points[:,d] + 1) * (priors[d, 1] - priors[d, 0]) / 2.) + priors[d,0]
    return sphere_points

def is_in_hypersphere(priors, params):
    """Returns whether or not the given params are within a hypersphere with edges defined by bounds"""

    # convert params to lay within the unit sphere
    unit_params = np.zeros_like(params)
    for d in range(priors.shape[0]):
        unit_params[d] = 2*(params[d] - priors[d,0]) / (priors[d,1] - priors[d,0]) - 1
    r = np.sqrt(np.sum(unit_params**2))
    if r >= 1: return False, r
    else:      return True, r

def organize_training_set(training_dir:str, train_frac:float, valid_frac:float, test_frac:float, 
                          param_dim, num_zbins, num_spectra, num_ells, k_dim, remove_old_files=True):
    """Takes a set of matrices and reorganizes them into training, validation, and tests sets
    
    Args:
        training_dir: Directory contaitning matrices to organize
        train_frac: Fraction of dataset to partition as the training set
        valid_frac: Fraction of dataset to partition as the validation set
        test_frac: Fraction of dataset to partition as the test set
        param_dim: Dimension of input parameter arrays
        mat_dim: Dimention of power spectra
        remove_old_files: If True, deletes old data files after loading data into \
            memory and before re-organizing. Default True.
    """
    all_filenames = next(os.walk(training_dir), (None, None, []))[2]  # [] if no file

    all_params = np.array([], dtype=np.int64).reshape(0,param_dim)
    all_galaxy_ps = np.array([], dtype=np.int64).reshape(0, num_spectra, num_zbins, k_dim, num_ells)
    all_nw_ps = np.array([], dtype=np.int64).reshape(0, 256)

    # load in all the data internally (NOTE: memory intensive!)
    # if "pk-raw.npz" in all_filenames:
    #     all_filenames = ["pk-raw.npz"]

    for file in all_filenames:
        if "pk-" in file:
            
            print("loading " + file + "...")
            F = np.load(training_dir+file)
            params = F["params"]
            galaxy_ps = F["galaxy_ps"]
            nw_ps = F["nw_ps"]
            del F
            all_params = np.vstack([all_params, params])
            all_galaxy_ps = np.vstack([all_galaxy_ps, galaxy_ps])
            all_nw_ps = np.vstack([all_nw_ps, nw_ps])

    N = all_params.shape[0]
    N_train = int(N * train_frac)
    N_valid = int(N * valid_frac)
    N_test = int(N * test_frac)
    assert N_train + N_valid + N_test <= N

    valid_start = N_train
    valid_end = N_train + N_valid
    test_end = N_train + N_valid + N_test
    assert test_end - valid_end == N_test
    assert valid_end - valid_start == N_valid

    if remove_old_files == True:
        for file in all_filenames:
            if "pk-" in file: os.remove(training_dir+file)

    print("splitting dataset into chunks of size [{:0.0f}, {:0.0f}, {:0.0f}]...".format(N_train, N_valid, N_test))

    np.savez(training_dir+"pk-training.npz", 
                params=all_params[0:N_train],
                galaxy_ps=all_galaxy_ps[0:N_train],
                nw_ps=all_nw_ps[0:N_train])
    np.savez(training_dir+"pk-validation.npz", 
                params=all_params[valid_start:valid_end], 
                galaxy_ps=all_galaxy_ps[valid_start:valid_end],
                nw_ps=all_nw_ps[valid_start:valid_end])
    np.savez(training_dir+"pk-testing.npz", 
                params=all_params[valid_end:test_end], 
                galaxy_ps=all_galaxy_ps[valid_end:test_end],
                nw_ps=all_nw_ps[valid_end:test_end]) 


def get_full_invcov(cov, num_zbins):

    invcov = torch.zeros_like(cov)
    for z in range(num_zbins):
        invcov[z] = torch.linalg.inv(cov[z])
    return invcov


def get_invcov_blocks(cov, num_spectra, num_zbins, num_kbins, num_ells):

    invcov_blocks = torch.zeros((num_spectra, num_zbins, num_ells*num_kbins, num_ells*num_kbins)).to(torch.float64)

    for z in range(num_zbins):
        for ps in range(num_spectra):
            cov_sub = cov[z, ps*num_ells*num_kbins: (ps+1)*num_ells*num_kbins,\
                             ps*num_ells*num_kbins: (ps+1)*num_ells*num_kbins]
            invcov_blocks[ps, z] = torch.linalg.inv(cov_sub)

            try:
                L = torch.linalg.cholesky(invcov_blocks[ps, z])
            except:
                print("ERROR!, matrix block [{:d}, {:d}, {:d}] is not positive-definite!".format(z, ps, ps))

    return invcov_blocks


def mse_loss(predict, target, invcov=None, normalized=False):
    return F.mse_loss(predict, target, reduction="sum")


def hyperbolic_loss(predict, target, invcov=None, normalized=False):
    return torch.mean(torch.sqrt(1 + 2*(predict - target)**2)) - 1


def hyperbolic_chi2_loss(predict, target, invcov, normalized=False):
    chi2 = delta_chi_squared(predict, target, invcov, normalized)
    return torch.mean(torch.sqrt(1 + 2*chi2)) - 1


def delta_chi_squared(predict, target, invcov, normalized=False):

    if not isinstance(predict, torch.Tensor):
        predict = torch.from_numpy(predict).to(torch.float32).to(invcov.device)
    if not isinstance(target, torch.Tensor):
        target = torch.from_numpy(target).to(torch.float32).to(invcov.device)

    if target.device != invcov.device:  target = target.to(invcov.device)
    if predict.device != invcov.device: predict = predict.to(invcov.device)

    # inputs are size [b, 1, nl*nk]
    # OR [nps, nz, nk, nl] (same as cosmo_inference)
    if predict.shape != target.shape:
        raise ValueError("ERROR! preidciton and target shape mismatch: "+ str(predict.shape) +", "+ str(target.shape))

    delta = predict - target

    chi2 = 0
    # calculate the delta chi2 for the entire emulator output, assuming normalization has been undone
    if normalized == False:
        if delta.dim() == 2:
            chi2 += torch.matmul(delta, torch.matmul(invcov, delta.unsqueeze(2)))
        elif delta.dim() == 4:
            (nps, nz, nk, nl) = delta.shape
            for z in range(nz):
                chi2 += torch.matmul(delta[:,z].flatten(), 
                        torch.matmul(invcov[z], 
                        delta[:,z].flatten()))
        else:
            raise ValueError("Expected input data with 2 or 5 dimensions, but got " + str(delta.dim()))
    else:
        assert len(delta.shape) == 2
        chi2 = torch.bmm(delta.unsqueeze(1), delta.unsqueeze(2)).squeeze()

    chi2 = torch.sum(chi2)
    return chi2


def calc_avg_loss(emulator, data_loader, loss_function, bin_idx=None, mode="galaxy_ps"):
    """run thru the given data set and returns the average loss value"""

    # if net_idx not specified, recursively call the function with all possible values
    if bin_idx == None and mode == "galaxy_ps":
        total_loss = torch.zeros(emulator.num_spectra, emulator.num_zbins, requires_grad=False)
        for (ps, z) in itertools.product(range(emulator.num_spectra), range(emulator.num_zbins)):
            total_loss[ps, z] = calc_avg_loss(emulator, data_loader, loss_function, [ps, z], mode)
        return total_loss
    
    emulator.galaxy_ps_model.eval()
    avg_loss = 0.
    with torch.no_grad():
        for (i, batch) in enumerate(data_loader):
            if mode == "galaxy_ps":
                params = emulator.galaxy_ps_model.organize_parameters(batch[0])
                params = normalize_cosmo_params(params, emulator.input_normalizations)
                prediction = emulator.galaxy_ps_model.forward(params, (bin_idx[1] * emulator.num_spectra) + bin_idx[0])
                target = torch.flatten(batch[1][:,bin_idx[0],bin_idx[1]], start_dim=1)
            elif mode == "nw_ps":
                params = batch[0][:,:emulator.nw_ps_model.input_dim]
                params = normalize_cosmo_params(params, emulator.input_normalizations[:,0,:emulator.input_dim])
                prediction = emulator.nw_ps_model.forward(params)
                target = batch[2]

            if emulator.normalization_type == "pca":
                prediction = pca_inverse_transform(prediction, emulator.principle_components, emulator.training_set_variance)
                avg_loss += loss_function(prediction, target, emulator.invcov_blocks[bin_idx], False).item()
            else:
                avg_loss += loss_function(prediction, target, emulator.invcov_blocks, True).item()

    return avg_loss / (len(data_loader.dataset))


def pca_transform(data, components, std):
    return (data / std) @ components.T


def pca_inverse_transform(reduced_data:torch.Tensor, components, std):
    if reduced_data.dim() == 2:
        reduced_data = reduced_data.unsqueeze(1)

    return (torch.matmul(reduced_data, components) * std).squeeze()


def normalize_cosmo_params(params, normalizations):
    return (params - normalizations[0]) / (normalizations[1] - normalizations[0])


# TODO: Move these to dataset.py
def normalize_power_spectrum(ps_raw, ps_fid, sqrt_eigvals, Q):

    # assumes ps has shape [b, nps, z, nk*nl]
    ps_new = torch.zeros_like(ps_raw)
    for (ps, z) in itertools.product(range(ps_new.shape[1]), range(ps_new.shape[2])):
        ps_new[:,ps, z] = ((ps_raw[:, ps, z] @ Q[ps, z]) - (ps_fid[ps, z].flatten() @ Q[ps, z])) * sqrt_eigvals[ps, z]
    return ps_new


def generate_PCs(ps_raw):

    print(ps_raw.shape)
    cov = torch.cov(ps_raw)


def un_normalize_power_spectrum(ps_raw, ps_fid, sqrt_eigvals, Q, Q_inv):
    """
    Reverses normalization of a batch of output power spectru based on the method developed by arXiv:2402.17716.

    Args:
        ps: () power spectrum to reverse normalization. Expected shape is either [nb, nps, nz, nk*nl] or [nb, 1, nk*nl]  
        ps_fid: fiducial power spectrum used to reverse normalization. Expected shape is [nps*nz, nk*nl]  
        sqrt_eigvals: square root eigenvalues of the inverse covariance matrix. Expected shape is [nps*nz, nk*nl]  
        Q: eigenvectors of the inverse covariance matrix. Expected shape is [nps*nz, nk*nl, nk*nl]  
        Q_inv: inverse eigenvectors of the inverse covariance matrix. Expected shape is [nps*nz, nk*nl, nk*nl]  
        net_idx: (optional) index specifying the specific sub-network output to reverse normalization. Default None. If not specified, will reverse normalization for the entire emulator output
    Returns:
        ps_new: galaxy power spectrum multipoles in units of (Mpc/h)^3 in the same shape as ps
    """

    ps_new = torch.zeros_like(ps_raw)
    # assumes shape is [b, nps, nz, nk*nl]
    if len(ps_raw.shape) == 4:
        for (ps, z) in itertools.product(range(ps_new.shape[1]), range(ps_new.shape[2])):
            ps_new[:, ps, z] = (ps_raw[:, ps, z] / sqrt_eigvals[ps, z] + (ps_fid[ps, z] @ Q[ps, z])) @ Q_inv[ps, z]
    # assumes shape is [nps, nz, nk*nl]
    elif len(ps_raw.shape) == 3:
        for (ps, z) in itertools.product(range(ps_new.shape[0]), range(ps_new.shape[1])):
            ps_new[ps, z] = (ps_raw[ps, z] / sqrt_eigvals[ps, z] + (ps_fid[ps, z] @ Q[ps, z])) @ Q_inv[ps, z]
    else:
        print("ERROR! Incorrect input shape for ps_raw!", ps_raw.shape)
        raise IndexError

    return ps_new