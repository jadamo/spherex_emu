# This script estimates a "covariance matrix" based on data in the training set
# NOTE: Use this as a substitute for if you don't have an actual covariance matrix

import torch
import spherex_emu.filepaths as filepaths
from spherex_emu.dataset import pk_galaxy_dataset
from spherex_emu.emulator import pk_emulator

def main():

    config_file = filepaths.network_pars_dir + "network_pars_2_sample_2_redshift.yaml"
    emulator = pk_emulator(config_file)

    train_data = emulator._load_data("training", 1., False)

    data = train_data.pk
    
    cov = torch.zeros((train_data.num_zbins, train_data.num_samples*train_data.num_ells*train_data.num_kbins,
                                             train_data.num_samples*train_data.num_ells*train_data.num_kbins))

    for z in range(train_data.num_zbins):
        print("Estimating cov for redshif bin", z)
        flat_data = data.view(-1,train_data.num_zbins,train_data.num_samples*train_data.num_ells*train_data.num_kbins)

        cov[z] = torch.cov(flat_data[:,z].T)
        #print(flat_data[:,z].T.shape)
    
        try:
            L = torch.linalg.cholesky(cov[z])
        except:
            print("WARNING: covariance is not positive definite!")
            
    print("Done!")
    torch.save(cov, filepaths.base_dir+emulator.config_dict["training_dir"]+"data_cov.dat")
if __name__ == "__main__":
    main()