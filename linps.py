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
import lhc
#from lhc import create_lhs_samples


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

lhc = open("lhc.txt", "r")
lhc = lhc.read().strip('[').strip(']').split()
print(lhc)

#lhc_array = np.array(eval(content))
#lhc_array = lhc_array.reshape(3,5)

#content = content.strip('[',).strip(']').strip('/n')
#lhc_list = list(map(float, content.split(',')))
#lhc_array = np.array(lhc_list).reshape(3,5)

#SET: x = # of data vectors, y = # of params

'''
x = 3
y = 5

f = open("lhc.txt", 'r')
fog = f.read()
f1 = fog.strip('[]')
f2 = np.fromstring(f1, sep=',')
f3 = np.reshape(f2, (x, y))
f4 = [list(row) for row in f3]
print(f4)
'''

with multiprocessing.Pool() as pool:
	results = pool.map(get_linps, lhc)

out_param, out_k, out_psm, out_psq = results

np.savez("/home/u14/gibbins/spherex_emu/out.npz",params=out_param,psm=out_psm,psq=out_psq)
