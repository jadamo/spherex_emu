import yaml, os
from scipy.stats import qmc
from torch.nn import functional as F
import numpy as np

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

def get_parameter_ranges(config_dict):
    """Organizes parameters found in the config dict into a standard form.
    TODO: Update to handle multiple redshift bins / tracers"""

    cosmo_params = dict(sorted(config_dict["cosmo_params"].items()))
    bias_params = dict(sorted(config_dict["bias_params"].items()))

    params_dict = {**cosmo_params, **bias_params}

    priors = np.array(list(params_dict.values()))
    params = list(params_dict.keys())
    return params, priors

def prepare_ps_inputs(sample, fiducial_params, num_tracers, num_zbins):
    """takes a set of parameters and oragnizes them to the format expected by ps_1loop"""
    param_vector = []
    # fill in cosmo params in the order ps_1loop expects
    for pname in list(fiducial_params["cosmo_params"].keys()):
        if pname in sample:
            #print(params.index(pname))
            param_vector.append(sample[pname])
        else:
            param_vector.append(fiducial_params["cosmo_params"][pname])

    # fill in bias params
    # The below is a (unrealistic) case in which all tracers have the same nuisance parameter values.
    for isample in range(num_tracers):
        for iz in range(num_zbins):
            sub_vector = []
            for pname in list(fiducial_params["bias_params"].keys()):
                if pname in sample:
                    sub_vector.append(sample[pname])
                else:
                    sub_vector.append(fiducial_params["bias_params"][pname])
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

def organize_training_set(training_dir:str, train_frac:float, valid_frac:float, test_frac:float, 
                          param_dim, num_zbins, num_tracers, k_dim, remove_old_files=True):
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
    all_pk = np.array([], dtype=np.int64).reshape(0, num_zbins, num_tracers, 2, k_dim)

    # load in all the data internally (NOTE: memory intensive!)    
    for file in all_filenames:
        if "pk-" in file:

            F = np.load(training_dir+file)
            params = F["params"]
            pk = F["pk"]
            del F

            all_params = np.vstack([all_params, params])
            all_pk = np.vstack([all_pk, pk])

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

def calc_avg_loss(net, data_loader):
    """run thru the given data set and returns the average loss value"""

    net.eval()
    avg_loss = 0.
    for (i, batch) in enumerate(data_loader):
        params = data_loader.dataset.get_repeat_params(batch[2], data_loader.dataset.num_samples, data_loader.dataset.num_zbins)
        prediction = net(params)
        avg_loss += F.mse_loss(prediction, batch[1], reduction="sum").item()

    avg_loss /= len(data_loader)
    return avg_loss

def normalize(X, normalizations):
    min_v = normalizations[0]
    max_v = normalizations[1]
    return (X - min_v) / (max_v - min_v)

def un_normalize(X, normalizations):
    min_v = normalizations[0]
    max_v = normalizations[1]
    return (X * (max_v - min_v)) + min_v
