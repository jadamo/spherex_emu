import numpy as np
import scipy.stats as sp # for calculating standard error
import camb, itertools
from camb import model
from math import comb

#from spherex_emu.fastpt import FASTPT
from spherex_emu.gpsclass import CalcGalaxyPowerSpec
import spherex_emu.filepaths as filepaths
from spherex_emu.utils import prepare_ps_inputs, load_config_file, fgrowth

#Creates linear power spectra from priors - input into galaxy ps class
def get_power_spectrum(params, kmin, kmax, num_kbins, num_samples, z_eff):

    params[0] = params[0] * 100
    num_spectra = num_samples + comb(num_samples, 2)
    
    num_cosmo_params = 5
    num_bias_params = 4
    pk = np.zeros((len(z_eff), num_spectra, 2, num_kbins))
    k = np.zeros((num_kbins)) #number of samples x number of k bins
    
    for z_idx in range(len(z_eff)):
        z = z_eff[z_idx]
        H0, ombh2, omch2, As, ns = params[0], params[1], params[2], params[3], params[4]
        pars = camb.CAMBparams()
        pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2)
        pars.InitPower.set_params(As=As, ns=ns)
        pars.set_matter_power(redshifts=[z], kmax=2.0) #sets redshift and mode for ps
        pars.NonLinear = model.NonLinear_none #set to be linear
        results = camb.get_results(pars)
        kh, z_camb, pk_lin = results.get_matter_power_spectrum(minkh=kmin, maxkh=kmax, npoints=num_kbins) #pk is 2 values 
        
        print(pk_lin[0])
        Om0 = results.get_Omega("cdm", 0) + results.get_Omega("baryon", 0)
        f = fgrowth(z, Om0)
        k = kh

        sample_idx = 0
        for isample1, isample2 in itertools.product(range(num_samples), repeat=2):
            if isample1 > isample2: continue

            idx1 = num_cosmo_params + (isample1*num_bias_params) + (z_idx * num_bias_params*num_samples)
            idx2 = num_cosmo_params + (isample2*num_bias_params) + (z_idx * num_bias_params*num_samples)
            bias1 = params[idx1:idx1+num_bias_params]
            bias2 = params[idx2:idx2+num_bias_params]

            nonlin = CalcGalaxyPowerSpec(f,pk_lin[0],kh,bias1,bias2,params[:num_cosmo_params])
            pk[z_idx, sample_idx, 0, :] = nonlin.get_nonlinear_ps(0)
            pk[z_idx, sample_idx, 1, :] = nonlin.get_nonlinear_ps(2)
            sample_idx+=1
    return k, pk

def main():
        
    cosmo_dict = load_config_file(filepaths.cosmo_pars_dir+"tns_2_sample_2_redshift.yaml")
    survey_pars = load_config_file(filepaths.survey_pars_dir + 'survey_pars_2_sample_2_redshift.yaml')

    ndens_table = np.array([[float(survey_pars['number_density_in_hinvMpc_%s' % (i+1)][j]) for j in range(survey_pars['nz'])] for i in range(survey_pars['nsample'])])
    num_samples = ndens_table.shape[0]
    z_eff = (np.array(survey_pars["zbin_lo"]) + np.array(survey_pars["zbin_hi"])) / 2.

    # TODO: Place this info in a better location like a survey pars file
    kmin = 0.01
    kmax = 0.25
    kbins = 26 # smallest possible is 4 and has to be even
    k = np.linspace(0.01, 0.25, 25)
    alternate_params = {}

    param_vector = prepare_ps_inputs(alternate_params, cosmo_dict, 2, len(z_eff))
    k, pk = get_power_spectrum(param_vector, kmin, kmax, kbins, num_samples, z_eff)
    
    np.save(filepaths.data_dir+"pk_tns_fid.npy", pk)

    # #Don't save output of get_linps to results, then to these arrays
    # #instead, I save the output directly to the arrays
    # #might not be the best idea but I got it to work
    # out_param = np.zeros((nsamples,prior[:,0].size))
    # out_k = np.zeros((nsamples,npoints))
    # out_psm = np.zeros((nsamples,npoints))
    # out_psq = np.zeros((nsamples,npoints))


    # #loop over the different combinations of tracers
    # #will find the power spectrum for every cosmology at a particular tracer combination
    # l=0
    # for j in range(ntracers):
    #     for k in range(j,ntracers):
    #         out_param[l,:], out_k[l,:], out_psm[l,:], out_psq[l,:] = get_linps(create_lhs_samples(x, prior),bias[j,:],bias[k,:],npoints)
    #         l+=1

    # #Sorry, I changed this
    # np.savez("/Users/anniemoore/desktop/out.npz",params=out_param,psm=out_psm,psq=out_psq)

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

if __name__ == "__main__":
    main()