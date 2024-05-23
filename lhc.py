import numpy as np
import pyDOE
from pyDOE import lhs

#Cosmo Parameters
prior = np.array([[20,100], #H0
                  [0.005,1], #omega_b
                  [0.1,.3], #omega_c
                  [1.2e-9,2.7e-9], #As
                  [0.8,1.2]]) #ns

#Number of cosmo params in prior array
n_dim = 5

#g: inputs number of wanted samples and prior array to output number of samples with randomly set cosmology from prior
def create_lhs_samples(n_samples , prior):
    lhs_samples = lhs(n_dim, n_samples) #creates lhc with values 0 to 1
    cosmo_samples = prior[:,0] + (prior[:,1] - prior[:,0]) * lhs_samples #scales each value to the given priors
    return cosmo_samples

np.savetxt('params.txt', create_lhs_samples(1,prior))
