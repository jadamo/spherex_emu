#import matplotlib
#import matplotlib.pyplot as plt
import numpy as np
#import pandas as pd
import scipy.stats as sp # for calculating standard error
import multiprocessing
from math import cos, exp, pi
from scipy.integrate import quad, quad_vec
from scipy import interpolate
from scipy import integrate
from scipy.special import eval_legendre
from fastpt import FASTPT
import camb
from camb import model
import pyDOE
from pyDOE import lhs
import time

from gpsclass import CalcGalaxyPowerSpec
from lhc import create_lhs_samples
#import out

#Galaxy Bias Parameter
bias = np.array([1.9,-0.6,(-4./7)*(1.9-1),(32./315.)*(1.9-1)])

#Cosmo Parameters
prior = np.array([[20,100], #H0
                  [0.005,1], #omega_b
                  [0.1,.3], #omega_c
                  [1.2e-9,2.7e-9], #As
                  [0.8,1.2]]) #ns

#Creates linear power spectra from priors - input into galaxy ps class
def get_linps(params):
    npoints = 10 #number of ps/k values: smallest possible is four & need to be even numbers
    psm = np.zeros((len(params[:,0]),npoints)) #number of samples x number of k bins
    psq = np.zeros((len(params[:,0]),npoints))
    k = np.zeros((len(params[:,0]),npoints)) #number of samples x number of k bins
    for row in range(len(params[:,0])):
        H0, ombh2, omch2, As, ns = params[row,0], params[row,1], params[row,2], params[row,3], params[row,4]
        pars = camb.CAMBparams()
        pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2)
        pars.InitPower.set_params(As=As, ns=ns)
        pars.set_matter_power(redshifts=[0.6], kmax=2.0) #sets redshift and mode for ps
        pars.NonLinear = model.NonLinear_none #set to be linear
        results = camb.get_results(pars)
        kh, z, pk = results.get_matter_power_spectrum(minkh=1e-3, maxkh=.2, npoints=npoints) #pk is 2 values 
        f = .7 #####PLACEHOLDER
        nonlin = CalcGalaxyPowerSpec(f,pk[0],kh,bias,params[row])
        ps_nonlin_mono = nonlin.get_nonlinear_ps(0)
        ps_nonlin_quad = nonlin.get_nonlinear_ps(2)
        k[row] = (kh)
        psm[row] = ps_nonlin_mono #(pk[0])
        psq[row] = ps_nonlin_quad #(pk[2])
    return params[row], k[0], psm[0], psq[0] #karray, Psnonlin = get_linps(params)

#Parallelizing linps

with multiprocessing.Pool() as pool:
	results = pool.map(get_linps, create_lhs_samples(x, prior))


#Number of PS to Generate
x = 1

out_param, out_k, out_psm, out_psq = results

np.save(out,params=out_param,psm=out_psm,psq=out_psq)

#prints parameters, mono ps, quad ps on new line in text file
#f = open("trainingset.txt", "a")
#f.write("\n")
#f.write(str(out_param))
#f.write(str(out_psm))
#f.write(str(out_psq))
#f.close()

#prints k modes into text file
#g = open("ktraining.txt", "w")
#g.write(str(out_k))
#g.close()
