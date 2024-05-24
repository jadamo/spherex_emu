import numpy as np
from math import exp
from scipy import integrate
from scipy.integrate import quad, quad_vec
from scipy.special import eval_legendre
from fastpt import FASTPT
import camb
from camb import model


#Calculates Galaxy PS from Linear PS
class CalcGalaxyPowerSpec:
    #modified input to take galaxy bias values from two tracer bins
    def __init__(self,fz,Plin,karray,galaxyBiasTracer1,galaxyBiasTracer2,cosmoParams):
        #Define linear matter power spectrum and k array generated from CAMB
        self.fz = fz
        self.P_lin  = Plin
        self.k_vec = karray
        
        self.sigv = 450.
        
        #pull out values of cosmology parameters used to generate the linear power spectrum from array
        self.H0, self.omb, self.omc, self.As, self.ns = cosmoParams
        
        #assign values of bias parameters from bias array
        #linear bias, quadratic bias, tidal bias, non-local bias

        #will save bias values from both tracer bins
        #self.b1a --> tracer1 and self.b1b --> tracer2
        self.b1a, self.b2a, self.bsa, self.b3nla = galaxyBiasTracer1
        self.b1b, self.b2b, self.bsb, self.b3nlb = galaxyBiasTracer2

        print(self.b1a)
        print(self.b1b)
        
        #computes FASTPT Grid. Needing for following power spectra calculations
        # initialize the FASTPT class
        # including extrapolation to higher and lower k
        self.fastpt=FASTPT(self.k_vec,to_do=['all'],low_extrap=-5,high_extrap=3,n_pad=500)
        
    def get_linear_kaiser(self,x):
        """Calculates the density-density, density-velocity (I call this cross)
           and velocity-velocity power spectra to third order"""
        
        #Define bias values
        #b1 = self.b1
        #b2 = self.b2
        #bs = self.bs
        #b3nl = self.b3nl
        
        ### DENSITY CALCULATION ###
        
        #calls function in FASTPT to compute all necessary components for density ps. stores them in one array
        PXXNL_b1b2bsb3nl_d = self.fastpt.one_loop_dd_bias_b3nl_density(self.P_lin, C_window=.65)
        
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
        Pgg_d = ((self.b1a*self.b1b) * (self.P_lin+one_loopkz_d)
               + 0.5*(self.b1a*self.b2b + self.b1b*self.b2a)*Pd1d2_d + 
               (1./4.)*(self.b2a*self.b2b)*(Pd2d2_d - 2.*sig4kz)
               + 0.5*(self.b1a*self.bsb+self.b1b*self.bsa)*Pd1s2_d + 
               (1./4.)*(self.b2a*self.bsb+self.b2b*self.bsa)*(Pd2s2_d - 4./3*sig4kz)
               + (1./4)*(self.bsa*self.bsb)*(Ps2s2_d - 8./9*sig4kz) + 0.5*(self.b1a*self.b3nlb+self.b1b*self.b3nla)*sig3nl)
        
        ### VELOCITY CALCULATION ###
        
        #calls function in FASTPT to compute all necessary components for velocity ps. stores them in one array
        PXXNL_b1b2bsb3nl_v = self.fastpt.one_loop_dd_bias_b3nl_velocity(self.P_lin, C_window=.65)

        #Breaks up earlier arrays into individual components
        #Some are the same as the density calculation. Only new terms are computed here
        [one_loopkz_v, Pd1d2_v, Pd1s2_v] = [
                np.outer(1, PXXNL_b1b2bsb3nl_v[0]),
                np.outer(1, PXXNL_b1b2bsb3nl_v[3]),
                np.outer(1, PXXNL_b1b2bsb3nl_v[4])]
        
        #velocity-velocity galaxy power spectrum
        #need factor of (a*67.*fz/c)**2 to make units work
        Pgg_v = (self.P_lin+(a*self.H0*self.fz/c)**2*one_loopkz_v)
        
        ### CROSS CALCULATION ###
        
        #calls function in FASTPT to compute all necessary components for cross ps. stores them in one array
        PXXNL_b1b2bsb3nl_c = self.fastpt.one_loop_dd_bias_b3nl_cross(self.P_lin, C_window=.65)
        
        #Breaks up earlier arrays into individual components
        #Some are the same as the earlier calculations. Only new terms are computed here
        [one_loopkz_c] = [np.outer(1, PXXNL_b1b2bsb3nl_c[0])]
        
        #density-velocity galaxy power spectrum
        #need factor of (a*67.*fz/c) to make units work
        Pgg_c =  (0.5*(self.b1a+self.b1b)*(self.P_lin+(a*self.H0*self.fz/c)*one_loopkz_c) 
                   + (a*self.H0*self.fz/c)*(0.5*(self.b2a+self.b2b)*Pd1d2_v + 0.5*(self.bsa+self.bsb)*Pd1s2_v) 
                   + 0.5*(self.b3nla+self.b3nlb)*sig3nl)
        
        #kaiser = Pgg_d + 2.*f*x**2*Pgg_c + f**2*x**4*Pgg_v
        
        #Combine terms we just calculated to get kaiser term 
        return Pgg_d + 2.*self.fz*x**2*Pgg_c + self.fz**2*x**4*Pgg_v #Pgg_d, Pgg_c, Pgg_v
    
    def get_tns_corrections(self,x):
        """Calculate the TNS correction terms. These account for nonlinearities arising from coupling
           between the density and velocity fields"""
        
        b1_sym = 0.5*(self.b1a+self.b1b)
        #Here I get the individual terms for the TNS corrections.
        A1, A3, A5, B0, B2, B4, B6, P_Ap1, P_Ap3, P_Ap5 = self.fastpt.RSD_components(self.P_lin, self.fz/b1_sym, P_window=None, C_window=0.65)
        
        #These are found in the FAST-PT github under RSD.py
        A_mu2 = self.k_vec*(self.fz/b1_sym)*(A1 + P_Ap1)
        A_mu4 = self.k_vec*(self.fz/b1_sym)*(A3 + P_Ap3) 
        A_mu6 = self.k_vec*(self.fz/b1_sym)*(A5 + P_Ap5)

        B_mu2 = ((self.fz/b1_sym)*self.k_vec)**2*B0
        B_mu4 = ((self.fz/b1_sym)*self.k_vec)**2*B2
        B_mu6 = ((self.fz/b1_sym)*self.k_vec)**2*B4
        B_mu8 = ((self.fz/b1_sym)*self.k_vec)**2*B6
        
        #return A_mu2, A_mu4, A_mu6, B_mu2, B_mu4, B_mu6, B_mu8
        return ((b1_sym)**3*A_mu2+(b1_sym)**4*B_mu2)*x**2 \
            + ((b1_sym)**3*A_mu4+(b1_sym)**4*B_mu4)*x**4 \
            + ((b1_sym)**3*A_mu6+(b1_sym)**4*B_mu6)*x**6 \
            + (b1_sym)**4*B_mu8*x**8 
    
    def get_FOG(self,x,sigv):
        """Describes effect of the velocity field on small scale galaxy clustering"""
        
        #FOG term can take different functional forms. Here we choose to use an exponential
        arg = self.fz*x*(self.k_vec/(self.H0/100))*self.sigv/(self.H0)
        
        return np.exp(-.5*arg**2)
    
    def get_integrand(self, x,pole):
        """Integrand needed to perform integral"""
        
        coeff = (2.*pole +1.)/2.
        
        return coeff*eval_legendre(pole,x)*self.get_FOG(x,self.sigv)*(self.get_linear_kaiser(x)+self.get_tns_corrections(x))
    
    def get_nonlinear_ps(self,pole):
        
        result = np.zeros((1,len(self.k_vec)))
        
        #def get_integrand(self, x):
        #    """Integrand needed to perform integral"""
        
        #    coeff = (2.*pole +1.)/2.
        
        #    return coeff*eval_legendre(pole,x)*self.get_FOG(x,self.sigv)*(self.get_linear_kaiser(x)+self.get_tns_corrections(x))

        
        result[0,:], error = integrate.quad_vec(self.get_integrand, -1, 1,args=(pole,))
            
        return result