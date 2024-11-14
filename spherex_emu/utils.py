import yaml, os
from scipy.stats import qmc
from scipy.special import hyp2f1
from torch.nn import functional as F
import numpy as np
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

    # cosmo_params = dict(sorted(cosmo_dict["cosmo_params"].items()))
    # bias_params = dict(sorted(cosmo_dict["bias_params"].items()))
    bias_params = {}
    for param in cosmo_dict["bias_params"]:
        if "prior" in cosmo_dict["bias_params"][param]:
            cosmo_params[param] = [cosmo_dict["bias_params"][param]["prior"]["min"],
                                   cosmo_dict["bias_params"][param]["prior"]["max"]]

    params_dict = {**cosmo_params, **bias_params}
    priors = np.array(list(params_dict.values()))
    params = list(params_dict.keys())
    return params, priors

def prepare_ps_inputs(sample, cosmo_dict, num_spectra, num_zbins):
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
    # The below is an (unrealistic) case in which all tracers have the same nuisance parameter values.
    for isample in range(num_spectra):
        for iz in range(num_zbins):
            sub_vector = []
            for pname in list(cosmo_dict["bias_param_names"]):
                key = pname+"_"+str(isample)+"_"+str(iz)

                if key in sample:
                    sub_vector.append(sample[key])
                elif pname in sample:
                    sub_vector.append(sample[pname])
                elif key in cosmo_dict["bias_params"]:
                    sub_vector.append(cosmo_dict["bias_params"][key]["value"])
                # special cases when bias parameters depend on other bias parameter values (TNS model)
                elif pname == "bs2" and cosmo_dict["bias_params"][pname]["value"] == -99:
                    sub_vector.append((-4./7)*(cosmo_dict["bias_params"]["b1"+"_"+str(isample)+"_"+str(iz)]["value"]-1))
                elif pname == "b3nl" and cosmo_dict["bias_params"][pname]["value"] == -99:
                    sub_vector.append((32./315)*(cosmo_dict["bias_params"]["b1"+"_"+str(isample)+"_"+str(iz)]["value"]-1))
                else:
                    sub_vector.append(cosmo_dict["bias_params"][pname]["value"])
            param_vector += sub_vector

    return np.array(param_vector)

def fgrowth(z,Om0):
    """Calculates the LambdaCDM growth rate f_growth(z, Om0)

    Args:
        z: cosmological redshift
        Om0: matter density parameter
    """
    return(1. + 6*(Om0-1)*hyp2f1(4/3., 2, 17/6., (1-1/Om0)/(1+z)**3)
                /( 11*Om0*(1+z)**3*hyp2f1(1/3., 1, 11/6., (1-1/Om0)/(1+z)**3) ))


def make_latin_hypercube(priors, N):
    """Generates a latin hypercube of N samples with lower and upper bounds given by priors"""

    n_dim = priors.shape[0]

    sampler = qmc.LatinHypercube(d=n_dim)
    params = sampler.random(n=N)

    for i in range(params.shape[1]):
        params[:,i] = (params[:,i] * (priors[i, 1] - priors[i, 0])) + priors[i,0]
    
    return params

def organize_training_set(training_dir:str, train_frac:float, valid_frac:float, test_frac:float, 
                          param_dim, num_zbins, num_tracers, num_ells, k_dim, remove_old_files=True):
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
    all_pk = np.array([], dtype=np.int64).reshape(0, num_zbins, num_tracers, num_ells, k_dim)

    # load in all the data internally (NOTE: memory intensive!)
    if "pk-raw.npz" in all_filenames:
        all_filenames = ["pk-raw.npz"]

    for file in all_filenames:
        if "pk-" in file:

            F = np.load(training_dir+file)
            params = F["params"]
            pk = F["pk"]
            del F
            all_params = np.vstack([all_params, params])
            all_pk = np.vstack([all_pk, pk])

    print(all_params.shape, all_pk.shape)
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
            if "CovA-" in file:
                os.remove(training_dir+file)

    print("splitting dataset into chunks of size [{:0.0f}, {:0.0f}, {:0.0f}]...".format(N_train, N_valid, N_test))

    np.savez(training_dir+"pk-training.npz", 
                params=all_params[0:N_train],
                pk=all_pk[0:N_train])
    np.savez(training_dir+"pk-validation.npz", 
                params=all_params[valid_start:valid_end], 
                pk=all_pk[valid_start:valid_end])
    np.savez(training_dir+"pk-testing.npz", 
                params=all_params[valid_end:test_end], 
                pk=all_pk[valid_end:test_end])    

def mse_loss(predict, target, invcov=None):
    return F.mse_loss(predict, target, reduction="sum")

def hyperbolic_loss(predict, target, invcov=None):
    return torch.mean(torch.sqrt(1 + 2*(predict - target)**2)) - 1

def hyperbolic_chi2_loss(predict, target, invcov):
    chi2 = delta_chi_squared(predict, target, invcov)
    return torch.mean(torch.sqrt(1 + 2*chi2)) - 1

def delta_chi_squared(predict, target, invcov):

    delta = predict - target # (b, nz, nps*nk*nl)
    if len(delta.shape) == 5:
        delta = torch.transpose(predict - target, 3, 4) # (b,nz,nps,nl,nk) -> (b, nz, nps, nk, nl)
        (_, nz, nps, nk, nl) = delta.shape
        delta = delta.reshape((-1, nz, nps*nk*nl)) # (nz, nps, nk, nl) --> (nz, nps*nk*nl) 

    delta_row = delta[:, :, None, :,] # (b, nz, 1, nps*nk*nl) 
    delta_col = delta[:, :, :, None,] # (b, nz, nps*nk*nl, 1) 

    # NOTE Matrix multiplication is for the last two indices; element wise for all other indices.
    chi2_component = torch.matmul(delta_row, torch.matmul(invcov, delta_col))[..., 0, 0] # invcov is (nz, nps*nk*nl, nps*nk*nl)
    chi2 = torch.sum(chi2_component)
    return chi2

def calc_avg_loss(net, data_loader, input_normalizations, 
                  ps_fid, invcov, eigvals, Q, Q_inv, loss_function):
    """run thru the given data set and returns the average loss value"""

    net.eval()
    avg_loss = 0.
    for (i, batch) in enumerate(data_loader):
        #params = data_loader.dataset.get_repeat_params(batch[2], data_loader.dataset.num_zbins, data_loader.dataset.num_samples)
        params = normalize_cosmo_params(batch[0], input_normalizations)
        prediction = net(params)
        prediction = un_normalize_power_spectrum(prediction, ps_fid, eigvals, Q, Q_inv)
        avg_loss += loss_function(prediction, batch[1], invcov).item()

    avg_loss /= len(data_loader)
    return avg_loss

def normalize_cosmo_params(params, normalizations):
    min_v = normalizations[0]
    max_v = normalizations[1]
    return (params - min_v) / (max_v - min_v)

def normalize_power_spectrum_diagonal(ps, ps_fid, inv_cov):
    ps_new = torch.zeros_like(ps)
    for z in range(ps_new.shape[1]):
        ps_new[:,z] = (ps[:,z] - ps_fid[z]) * torch.sqrt(torch.diag(inv_cov[z]))
                       
    return ps_new

def normalize_power_spectruml(ps, ps_fid, eigvals, Q):
    ps_new = torch.zeros_like(ps)
    for z in range(ps_new.shape[1]):
        ps_new[:,z] = ((ps[:, z].flatten() @ Q) - (ps_fid[z].flatten() @ Q)) * torch.sqrt(eigvals)         
    return ps_new

def un_normalize_power_spectrum_diagonal(ps, ps_fid, inv_cov):
    """
    Reverses normalization of a batch of output power spectru based on the method developed by Evan,
    assuming a totally diagonal covariance matrix
    NOTE: This funciton is in the process of being tested / deprecated!
    
    Args:
        ps: power spectrum to reverse normalization. Expected shape is [nb, nz, ns*nk*nl]
        ps_fid: fiducial power spectrum used to reverse normalization. Expected shape is [nb, nz, ns*nk*nl]
        inv_cov: (diagonal) inverse data covariance matrix used to reverse normalization. Expected shape is [nz, ns*nk*nl]
    Returns:
        ps_new: galaxy power spectrum in units of (Mpc/h)^3 in the same shape as ps
    """
    ps_new = torch.zeros_like(ps)
    for z in range(ps_new.shape[1]):
        ps_new[:,z] = (ps[:,z] / torch.sqrt(torch.diag(inv_cov[z]))) + ps_fid[z]
                       
    return ps_new

def un_normalize_power_spectrum(ps, ps_fid, eigvals, Q, Q_inv):
    """
    Reverses normalization of a batch of output power spectru based on the method developed by Evan.

    Args:
        ps: power spectrum to reverse normalization. Expected shape is [nb, nz, ns*nk*nl]
        ps_fid: fiducial power spectrum used to reverse normalization. Expected shape is [nz, ns*nk*nl]
        eigvals: eigenvalues of the inverse covariance matrix
        Q: eigenvectors of the inverse covariance matrix
        Q_inv: inverse eigenvectors of the inverse covariance matrix
    Returns:
        ps_new: galaxy power spectrum in units of (Mpc/h)^3 in the same shape as ps
    """
    ps_new = torch.zeros_like(ps)
    for z in range(ps_new.shape[1]):
        ps_new[:,z] = (ps[:,z] / torch.sqrt(eigvals[z]) + (ps_fid[z].flatten() @ Q[z])) @ Q_inv[z]
    
    #ps_new = (ps / torch.sqrt(eigvals) + (ps_fid @ Q))# @ Q_inv
    #print(ps_fid.shape, Q.shape, (ps_fid @ Q).shape)
    #print(ps_new.shape, eigvals.shape, ps_fid.shape, Q.shape)
    return ps_new

# TODO: Remove or rename function
def un_normalize(X, normalizations):
    min_v = normalizations[0]
    max_v = normalizations[1]
    return (X * (max_v - min_v)) + min_v
