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
import camb #/home/u14/gibbins/spherex_emu
from camb import model
import pyDOE
from pyDOE import lhs
import time

from gpsclass import CalcGalaxyPowerSpec
#import lhc
from lhc import create_lhs_samples


#Galaxy Bias Parameter
bias = np.array([[1.9,-0.6,(-4./7)*(1.9-1),(32./315.)*(1.9-1)],
                [1.9,-0.6,(-4./7)*(1.9-1),(32./315.)*(1.9-1)]])

ntracers = 1

nsamples = int(ntracers*(ntracers+1)/2)

print(nsamples)

npoints = 10

#Cosmo Parameters
prior = np.array([[20,100], #H0
                  [0.005,1], #omega_b
                  [0.1,.3], #omega_c
                  [1.2e-9,2.7e-9], #As
                  [0.8,1.2]]) #ns

#Creates linear power spectra from priors - input into galaxy ps class
def get_linps(params,bias1,bias2,npoints):
    #npoints = 10 #number of ps/k values: smallest possible is four & need to be even numbers
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
        nonlin = CalcGalaxyPowerSpec(f,pk[0],kh,bias1,bias2,params[row])
        ps_nonlin_mono = nonlin.get_nonlinear_ps(0)
        ps_nonlin_quad = nonlin.get_nonlinear_ps(2)
        k[row] = (kh)
        psm[row] = ps_nonlin_mono #(pk[0])
        psq[row] = ps_nonlin_quad #(pk[2])
    return params[row], k[0], psm[0], psq[0] #karray, Psnonlin = get_linps(params)

#Number of PS to Generate
x = 1

#Parallelizing linps

# with multiprocessing.Pool() as pool:
# 	results = pool.map(get_linps, create_lhs_samples(x, prior))

params_test = np.array([[6.22954856e+01, 5.88608308e-01, 1.78577808e-01, 2.01059374e-09, 1.11635806e+00]])
# print((create_lhs_samples(x, prior)))
# results = get_linps(create_lhs_samples(x, prior))

out_param = np.zeros((nsamples,prior[:,0].size))
out_k = np.zeros((nsamples,npoints))
out_psm = np.zeros((nsamples,npoints))
out_psq = np.zeros((nsamples,npoints))

l=0
for j in range(ntracers):
    for k in range(j,ntracers):
        out_param[l,:], out_k[l,:], out_psm[l,:], out_psq[l,:] = get_linps(params_test,bias[j,:],bias[k,:],npoints)
        l+=1


print(print(out_psm))

#out_param, out_k, out_psm, out_psq = results

np.savez("/Users/anniemoore/desktop/out.npz",params=out_param,psm=out_psm,psq=out_psq)

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
