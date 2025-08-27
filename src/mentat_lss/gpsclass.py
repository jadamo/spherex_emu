import numpy as np
from scipy import integrate
from scipy.special import eval_legendre
from mentat_lss.fastpt import FASTPT
from scipy.interpolate import InterpolatedUnivariateSpline
import camb
from camb import model
from mentat_lss.utils import fgrowth

#Calculates Galaxy PS from Linear PS
class CalcGalaxyPowerSpec:

    def __init__(self,zarray,karray,cosmoParams):
        """
        initializes the model for calculating galaxy power spectrum multipoles using the TNS model (https://arxiv.org/pdf/1006.0699).
        This class has been modified to take galaxy bias values from two tracer bins

        Args:
            zarray: np array with effective redshift values
            karray: desired kbin (centers) to calculate the power spectrum for
            cosmoParams: numpy array of cosmology parameters (H0, ombh2, omch2, As, ns)
        """

        #Define linear matter power spectrum generated from CAMB
        self.interp_k_len = 40
        self.set_k_arrays(karray)
        self.set_pk_lin(cosmoParams, zarray)
        self.fz = fgrowth(zarray, self.Om0)

        #pull out values of cosmology parameters used to generate the linear power spectrum from array
        self.H0, self.omb, self.omc, self.As, self.ns = cosmoParams

        #computes FASTPT Grid. Needing for following power spectra calculations
        # initialize the FASTPT class
        # including extrapolation to higher and lower k
        self.fastpt=FASTPT(self.k_fastpt,to_do=['all'],low_extrap=-5,high_extrap=3,n_pad=500)
        
    def set_k_arrays(self, k):
        """Sets both the desired output k-array, and an internal k-array to pass through camb and FASTPT"""
        dk = k[1] - k[0]
        self.kmin = k[0] - (dk / 2.)
        self.kmax = k[-1] + (dk / 2.)

        if len(k) > self.interp_k_len:
            raise ValueError("ERROR! more k-bins than calculated with fastpt requested! Either reduce # of k-bins or increase interp_k_len")

        self.k_output = k
        self.k_fastpt = np.geomspace(self.kmin, self.kmax, self.interp_k_len)

    def set_pk_lin(self, params, z_eff):
        """Calculates the linear matter power spectrum from camb"""

        H0, ombh2, omch2, As, ns = params[0], params[1], params[2], params[3], params[4]
        pars = camb.CAMBparams()
        pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2)
        pars.InitPower.set_params(As=As, ns=ns)
        pars.set_matter_power(redshifts=z_eff, kmax=2.0) #sets redshift and mode for ps
        pars.NonLinear = model.NonLinear_none #set to be linear
        results = camb.get_results(pars)
        kh, z_camb, self.pk_lin = results.get_matter_power_spectrum(minkh=self.kmin, maxkh=self.kmax, npoints=len(self.k_fastpt)) #pk is 2 values 
        self.Om0 = results.get_Omega("cdm", 0) + results.get_Omega("baryon", 0)
        
    def get_linear_kaiser(self,x, z_idx):
        """Calculates the density-density, density-velocity (I call this cross)
           and velocity-velocity power spectra to third order"""
        
        #Define bias values
        #b1 = self.b1
        #b2 = self.b2
        #bs = self.bs
        #b3nl = self.b3nl
        
        ### DENSITY CALCULATION ###
        
        #calls function in FASTPT to compute all necessary components for density ps. stores them in one array
        PXXNL_b1b2bsb3nl_d = self.fastpt.one_loop_dd_bias_b3nl_density(self.pk_lin[z_idx], C_window=.65)
        
        #Breaks up earlier array into individual components
        [one_loopkz_d, Pd1d2_d, Pd2d2_d, Pd1s2_d, Pd2s2_d, Ps2s2_d, sig4kz, sig3nl] = [
                np.outer(1, PXXNL_b1b2bsb3nl_d[0]),
                np.outer(1, PXXNL_b1b2bsb3nl_d[2]),
                np.outer(1, PXXNL_b1b2bsb3nl_d[3]),
                np.outer(1, PXXNL_b1b2bsb3nl_d[4]),
                np.outer(1, PXXNL_b1b2bsb3nl_d[5]),
                np.outer(1, PXXNL_b1b2bsb3nl_d[6]),
                np.outer(1, PXXNL_b1b2bsb3nl_d[7]),
                np.outer(1, PXXNL_b1b2bsb3nl_d[8])]
        
        a=1/(1+.61)
        c=299726.

        #density-density galaxy power spectrum
        #For multi-tracer, need to include bias from both bins when computing the galaxy ps
        #Ex: $\delta_{g,a}$ = b1a * $\delta^(1)$ + b2a/2. * $\delta^(2)$ + ...
        #    $\delta_{g,b}$ = b1b * $\delta^(1)$ + b2b/2. * $\delta^(2)$ + ...
        # So < $\delta_{g,a}\delta_{g,b}$ >  = b1a*b1b $\delta^(1)\delta^(1)$ + b1a
        Pgg_d = ((self.b1a*self.b1b) * (self.pk_lin[z_idx]+one_loopkz_d)
               + 0.5*(self.b1a*self.b2b + self.b1b*self.b2a)*Pd1d2_d + 
               (1./4.)*(self.b2a*self.b2b)*(Pd2d2_d - 2.*sig4kz)
               + 0.5*(self.b1a*self.bsb+self.b1b*self.bsa)*Pd1s2_d + 
               (1./4.)*(self.b2a*self.bsb+self.b2b*self.bsa)*(Pd2s2_d - 4./3*sig4kz)
               + (1./4)*(self.bsa*self.bsb)*(Ps2s2_d - 8./9*sig4kz) + 0.5*(self.b1a*self.b3nlb+self.b1b*self.b3nla)*sig3nl)
        
        ### VELOCITY CALCULATION ###
        
        #calls function in FASTPT to compute all necessary components for velocity ps. stores them in one array
        PXXNL_b1b2bsb3nl_v = self.fastpt.one_loop_dd_bias_b3nl_velocity(self.pk_lin[z_idx], C_window=.65)

        #Breaks up earlier arrays into individual components
        #Some are the same as the density calculation. Only new terms are computed here
        [one_loopkz_v, Pd1d2_v, Pd1s2_v] = [
                np.outer(1, PXXNL_b1b2bsb3nl_v[0]),
                np.outer(1, PXXNL_b1b2bsb3nl_v[3]),
                np.outer(1, PXXNL_b1b2bsb3nl_v[4])]
        
        #velocity-velocity galaxy power spectrum
        #need factor of (a*67.*fz/c)**2 to make units work
        Pgg_v = (self.pk_lin[z_idx]+(a*self.H0*self.fz[z_idx]/c)**2*one_loopkz_v)
        
        ### CROSS CALCULATION ###
        
        #calls function in FASTPT to compute all necessary components for cross ps. stores them in one array
        PXXNL_b1b2bsb3nl_c = self.fastpt.one_loop_dd_bias_b3nl_cross(self.pk_lin[z_idx], C_window=.65)
        
        #Breaks up earlier arrays into individual components
        #Some are the same as the earlier calculations. Only new terms are computed here
        [one_loopkz_c] = [np.outer(1, PXXNL_b1b2bsb3nl_c[0])]
        
        #density-velocity galaxy power spectrum
        #need factor of (a*67.*fz/c) to make units work
        Pgg_c =  (0.5*(self.b1a+self.b1b)*(self.pk_lin[z_idx]+(a*self.H0*self.fz[z_idx]/c)*one_loopkz_c) 
                   + (a*self.H0*self.fz[z_idx]/c)*(0.5*(self.b2a+self.b2b)*Pd1d2_v + 0.5*(self.bsa+self.bsb)*Pd1s2_v) 
                   + 0.5*(self.b3nla+self.b3nlb)*sig3nl)
        
        #kaiser = Pgg_d + 2.*f*x**2*Pgg_c + f**2*x**4*Pgg_v
        
        #Combine terms we just calculated to get kaiser term 
        return Pgg_d + 2.*self.fz[z_idx]*x**2*Pgg_c + self.fz[z_idx]**2*x**4*Pgg_v #Pgg_d, Pgg_c, Pgg_v
    
    def get_tns_corrections(self,x, z_idx):
        """Calculate the TNS correction terms. These account for nonlinearities arising from coupling
           between the density and velocity fields"""
        
        b1_sym = 0.5*(self.b1a+self.b1b)
        #Here I get the individual terms for the TNS corrections.
        A1, A3, A5, B0, B2, B4, B6, P_Ap1, P_Ap3, P_Ap5 = self.fastpt.RSD_components(self.pk_lin[z_idx], self.fz[z_idx]/b1_sym, P_window=None, C_window=0.65)
        
        #These are found in the FAST-PT github under RSD.py
        A_mu2 = self.k_fastpt*(self.fz[z_idx]/b1_sym)*(A1 + P_Ap1)
        A_mu4 = self.k_fastpt*(self.fz[z_idx]/b1_sym)*(A3 + P_Ap3) 
        A_mu6 = self.k_fastpt*(self.fz[z_idx]/b1_sym)*(A5 + P_Ap5)

        B_mu2 = ((self.fz[z_idx]/b1_sym)*self.k_fastpt)**2*B0
        B_mu4 = ((self.fz[z_idx]/b1_sym)*self.k_fastpt)**2*B2
        B_mu6 = ((self.fz[z_idx]/b1_sym)*self.k_fastpt)**2*B4
        B_mu8 = ((self.fz[z_idx]/b1_sym)*self.k_fastpt)**2*B6
        
        #return A_mu2, A_mu4, A_mu6, B_mu2, B_mu4, B_mu6, B_mu8
        return ((b1_sym)**3*A_mu2+(b1_sym)**4*B_mu2)*x**2 \
            + ((b1_sym)**3*A_mu4+(b1_sym)**4*B_mu4)*x**4 \
            + ((b1_sym)**3*A_mu6+(b1_sym)**4*B_mu6)*x**6 \
            + (b1_sym)**4*B_mu8*x**8 
    
    def get_FOG(self,x, z_idx):
        """Describes effect of the velocity field on small scale galaxy clustering"""
        
        #FOG term can take different functional forms. Here we choose to use an exponential
        arg = self.fz[z_idx]*x*(self.k_fastpt/(self.H0/100))*np.sqrt(self.sigva*self.sigvb)/(self.H0)
        
        return np.exp(-.5*arg**2)
    
    def get_integrand(self, x,pole, z_idx):
        """Integrand needed to perform integral"""
        
        coeff = (2.*pole +1.)/2.
        
        return coeff*eval_legendre(pole,x)*self.get_FOG(x, z_idx)*(self.get_linear_kaiser(x, z_idx)+self.get_tns_corrections(x, z_idx))
    
    def get_nonlinear_ps(self,z_idx:int, pole:int, galaxyBiasTracer1,galaxyBiasTracer2):
        """
        Args:
            z_idx: redshift bin to calculate for
            pole: multipole to calculate
            galaxyBiasTracer1: numpy array of galaxy bias parameters for the 1st tracer (b1, b2, bs2, b3nl, sigv)
            galaxyBiasTracer2: numpy array of galaxy bias parameters for the 2nd tracer (b1, b2, bs2, b3nl, sigv)
        """
        #will save bias values from both tracer bins
        #self.b1a --> tracer1 and self.b1b --> tracer2
        self.b1a, self.b2a, self.bsa, self.b3nla, self.sigva = galaxyBiasTracer1
        self.b1b, self.b2b, self.bsb, self.b3nlb, self.sigvb = galaxyBiasTracer2

        pk = np.zeros((1, len(self.k_fastpt)))
        
        #def get_integrand(self, x):
        #    """Integrand needed to perform integral"""
        
        #    coeff = (2.*pole +1.)/2.
        
        #    return coeff*eval_legendre(pole,x)*self.get_FOG(x,self.sigv)*(self.get_linear_kaiser(x)+self.get_tns_corrections(x))

        pk[0,:], error = integrate.quad_vec(self.get_integrand, -1, 1,args=(pole,z_idx))
        pk = self.interpolate_to_output_k(pk[0])

        return pk
    
    def interpolate_to_output_k(self, pk):
        pk_func = InterpolatedUnivariateSpline(self.k_fastpt, pk)
        return pk_func(self.k_output)