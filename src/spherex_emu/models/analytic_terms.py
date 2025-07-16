import numpy as np
from scipy.integrate import romb
from scipy.special import lpmv
from scipy.interpolate import InterpolatedUnivariateSpline, RectBivariateSpline
import itertools, math
import symbolic_pofk.linear as linear
from spherex_emu.cosmo_utils import LCDMCosmology, IRResum, get_log_extrap
import time

class analytic_eft_model():
    """Class contatining calculations for the non-emulated terms of the EFT galaxy power spectrum"""

    param_names_cosmo = ['h', 'omega_b', 'omega_c', 'As', 'ns', 'fnl']
    param_names_bias = ['b1', 'b2', 'bG2', 'bGamma3', 'bphi', 'bphid']
    param_names_ctr = ['c0', 'c2', 'c4', 'cfog']
    param_names_stoch = ['P_shot', 'a0', 'a2']
    param_names_stoch_cross = ['P_shot_cross', 'a0_cross', 'a2_cross']

    def __init__(self, num_tracers, redshift_list, ells, k):

        self.cosmology = LCDMCosmology(0.7, 0.3)

        self.redshift_list = redshift_list
        self.num_tracers = num_tracers
        self.num_spectra = self.num_tracers + math.comb(self.num_tracers, 2)
        self.ells = ells
        self.k = k
        self.mu = np.linspace(0.,1.,2**5+1)
        self.dmu = self.mu[1] - self.mu[0]

        self.k_lin = np.geomspace(1e-4, 10., 1000)
        self.sigma_v = 0.

    def set_ir_resum_params(self, h, D):
        # object for precalculated IR resummation stuff
        self.irres = IRResum(self.get_pk_lin, hubble=h, rbao=110., kwarg={"D" : D})
        # BAO damping factors
        #t = time.time()
        self.Sigma2 = self.irres.get_Sigma2(ks=0.2)
        self.dSigma2 = self.irres.get_dSigma2(ks=0.2)
        #print("Sigma2 takes {:0.1f}ms to calculate".format((time.time()-t)*1000))

    def set_params(self, param_vector):
        params = {}

        # create a dictionary of parameters for a given (very long) parameter vector. 
        # NOTE: it should be adjusted when the order of parameters in the parameter vector is changed.
        i = 0
        for pname in self.param_names_cosmo:
            params[pname] = param_vector[i]
            i += 1
        for isample in range(self.num_tracers):
            for iz in range(len(self.redshift_list)):
                for pname in self.param_names_bias + self.param_names_ctr + self.param_names_stoch:# + param_names_stoch_cross:
                    params['%s_%s_%s' % (pname, isample, iz)] = param_vector[i]
                    i += 1
        
        # other cosmological parameters
        params['Omega_b'] = params['omega_b'] / params['h']**2
        params['Omega_c'] = params['omega_c'] / params['h']**2
        params['Omega_m'] = params['Omega_b'] + params['Omega_c'] # no massive neutrinos assumed in symbolic_pofk
        params['sigma8'] = linear.As_to_sigma8(params['As'] * 1e9, params['Omega_m'], params['Omega_b'], params['h'], params['ns'])
        params['k_pivot'] = 0.05 # in unit of 1/Mpc

        # set reference cosmology
        self.reference_cosmology = LCDMCosmology(params["h"], 0.3)

        # linear growth factor & linear growth rate
        self.cosmology.set_Omega_m0(params["Omega_m"])
        self.cosmology.set_h(params["h"])

        # self._cosmo = LCDMCosmology(params['h'], params['Omega_m'])
        params['Dgrowth'] = np.array([self.cosmology.get_Dgrowth(redshift) for redshift in self.redshift_list]) / self.cosmology.get_Dgrowth(0.)
        params['fgrowth'] = np.array([self.cosmology.get_fgrowth(redshift) for redshift in self.redshift_list])

        # angular diameter distance & Hubble rate
        DA_list = np.array([self.cosmology.get_D_angular_in_h_inv_Mpc(redshift) for redshift in self.redshift_list])
        Hz_list = np.array([self.cosmology.get_E(redshift) for redshift in self.redshift_list])
        DA_ref_list = np.array([self.reference_cosmology.get_D_angular_in_h_inv_Mpc(redshift) for redshift in self.redshift_list])
        Hz_ref_list = np.array([self.reference_cosmology.get_E(redshift) for redshift in self.redshift_list])
    
        # Alcock-Paczynski effect parameters
        params['alpha_perp'] = DA_list / DA_ref_list
        params['alpha_para'] = Hz_ref_list / Hz_list

        return params

    def calculate_pk_lin(self, k, params):
 
        pk_lin = linear.plin_emulated(k, params['sigma8'], params['Omega_m'], params['Omega_b'], params['h'], params['ns'], emulator='fiducial', extrapolate=True)
        k_extrap, pk_extrap = get_log_extrap(k, pk_lin, xmin=1e-7, xmax=1e7)
        self.pk_lin_spl = InterpolatedUnivariateSpline(np.log(k_extrap), np.log(pk_extrap))

    def get_pk_lin(self, k, D, khigh=None): # in unit of [h^{-3} Mpc^3]
        pk_lin = np.exp(self.pk_lin_spl(np.log(k))) * D**2
        if khigh != None:
            pk_lin = pk_lin * np.exp(-(k / khigh))
        return pk_lin

    def get_pk_lin_irres_rsd(self, k, mu, f, D):
        # wiggly-non-wiggly decomposition
        plin = self.get_pk_lin(k, D)
        plin_nw = self.irres.get_pk_nw(k) * D**2
        plin_w = plin - plin_nw

        # BAO damping factor in redshift space
        Sigma2_1 = (1 + mu**2 * f * (2 + f)) * self.Sigma2
        Sigma2_2 = f**2 * mu**2 * (mu**2 - 1) * self.dSigma2
        Sigma2_tot = Sigma2_1 + Sigma2_2

        plin_nw_tile = np.tile(plin_nw, (len(mu),1)).T
        plin_w_tile = np.tile(plin_w, (len(mu),1)).T
        damp_fac = np.kron(k**2, D**2 * Sigma2_tot).reshape(len(k),len(mu))

        pk = plin_nw_tile + np.exp(-damp_fac) * plin_w_tile
        return pk

    def get_damping_factor(self, k, mu, f):
        kmu = np.kron(k, mu).reshape(len(k), len(mu))
        return np.exp(-0.5*(f * self.sigma_v*kmu)**2)

    def get_tree_term(self, k, mu, bias1, bias2, f, D):

        plin = self.get_pk_lin(k, D)
        plin_nw = self.irres.get_pk_nw(k) * D**2
        plin_w = plin - plin_nw

        # BAO damping factor in redshift space
        Sigma2_1 = (1 + mu**2 * f * (2 + f)) * self.Sigma2
        Sigma2_2 = f**2 * mu**2 * (mu**2 - 1) * self.dSigma2
        Sigma2_tot = Sigma2_1 + Sigma2_2

        plin_nw_tile = np.tile(plin_nw, (len(mu),1)).T
        plin_w_tile = np.tile(plin_w, (len(mu),1)).T
        damp_fac = np.kron(k**2, D**2 * Sigma2_tot).reshape(len(k),len(mu))

        Z1_tile1 = np.tile(bias1["b1"] + f * mu**2, (len(k), 1))
        Z1_tile2 = np.tile(bias2["b1"] + f * mu**2, (len(k), 1))

        return Z1_tile1 * Z1_tile2 * (plin_nw_tile + (1 + damp_fac) * np.exp(-damp_fac) * plin_w_tile)
    
    def get_ctr_terms(self, k, mu, bias1, bias2, ctr1, ctr2, f, D):

        plin_tile = self.get_pk_lin_irres_rsd(k, mu, f, D)
        #plin_tile = np.tile(plin, (len(mu),1)).T
        ctr_LO = ( ctr1["c0"] + ctr2["c0"])/2. + \
                 ((ctr1["c2"] + ctr2["c2"])/2. * f * mu**2) + \
                 ((ctr1["c4"] + ctr2["c4"])/2. * f**2 * mu**4)
        ctr_LO = -2 * np.kron(k**2, ctr_LO).reshape(len(k),len(mu)) * plin_tile

        ctr_NLO = (ctr1["cfog"] + ctr2["cfog"])/2. * f**4 * mu**4 * (bias1["b1"] + f * mu**2)* (bias2["b1"] + f * mu**2)
        ctr_NLO = - np.kron(k**4, ctr_NLO).reshape(len(k),len(mu)) * plin_tile

        return ctr_LO + ctr_NLO

    # def get_pkmu_for_ctr1_ref(self, k_ref, mu_ref, alpha_perp, alpha_para, irres=True):
    #     k_ref = np.atleast_1d(k_ref)
    #     mu_ref = np.atleast_1d(mu_ref)

    #     # mapping of (k, mu)
    #     fac = np.sqrt(1 + mu_ref**2 * ((alpha_perp / alpha_para)**2 - 1))
    #     mu = mu_ref * (alpha_perp / alpha_para) / fac
    #     k = np.kron(k_ref, fac).reshape(len(k_ref), len(mu_ref)) / alpha_perp

        # spline interpolation of mu^l k^2 P_lin(k)
    #     k_grid = np.geomspace(np.max([np.min(k), self._kmin_fft]), np.min([np.max(k), self._kmax_fft]), self._nmax_fft)
    #     mu_grid = np.linspace(0., 1., 51)
    #     # k2mul = np.kron(k_grid**2, mu_grid**l).reshape(len(k_grid), len(mu_grid))
    #     if irres:
    #         pkmu_grid = self.get_pk_lin_irres_rsd(k_grid, mu_grid)
    #     else:
    #         pkmu_grid = np.tile(self.get_pk_lin(k_grid), (len(mu_grid),1)).T

    #     pkmu_interp = RectBivariateSpline(k_grid, mu_grid, pkmu_grid)
    #     pkmu = pkmu_interp.ev(k, np.tile(mu, (len(k_ref), 1)))
    #     pkmu = pkmu / (alpha_perp**2 * alpha_para)

    #     if len(k_ref) == 1 or len(mu_ref) == 1:
    #         pkmu = np.ravel(pkmu)
    #     return pkmu
    
    # def get_pk_ell_ctr1_ref(self, k_ref, ells, alpha_perp, alpha_para, irres=True):
    #     k_ref = np.atleast_1d(k_ref)
    #     mu_ref = np.linspace(0.,1.,2**8+1)
    #     dmu = mu_ref[1] - mu_ref[0]

    #     coeffs = np.array([self.get_coeff_ctr1_multipole(l) for l in ells])
    #     coeffs = np.tile(coeffs, (len(k_ref), 1)).T

    #     # mapping of (k, mu)
    #     fac = np.sqrt(1 + mu_ref**2 * ((alpha_perp / alpha_para)**2 - 1))
    #     mu = mu_ref * (alpha_perp / alpha_para) / fac
    #     k = np.kron(k_ref, fac).reshape(len(k_ref), len(mu_ref)) / alpha_perp

    #     pkmu_ref = self.get_pkmu_for_ctr1_ref(k_ref, mu_ref, alpha_perp, alpha_para, irres=irres)

    #     pkmu_ref = np.tile(pkmu_ref, (len(ells),1,1))
    #     legendre = np.array([np.tile((2*l+1) * lpmv(0,l,mu_ref) * mu**l * self.fgrowth**(l/2), (len(k_ref),1)) * k**2 for l in ells])
    #     pk_ell_ctr1 = - 2 * romb(pkmu_ref * legendre, dx=dmu, axis=2)
    #     pk_ell_ctr1 = coeffs * pk_ell_ctr1

    #     return pk_ell_ctr1

    def get_stochastic_terms(self):
        return 0.

    def get_analytic_terms(self, param_vector):
        params = self.set_params(param_vector)
        
        self.calculate_pk_lin(self.k_lin, params)
        self.set_ir_resum_params(params["h"], 1.)

        pk_ell = np.zeros((len(self.redshift_list), self.num_spectra, len(self.ells), len(self.k)))
        for z in range(len(self.redshift_list)):

            # map (k, mu) to different values based on AP effect
            fac = np.sqrt(1 + self.mu**2 * ((params["alpha_perp"][z] / params["alpha_para"][z])**2 - 1))
            mu_eval = self.mu * (params["alpha_perp"][z] / params["alpha_para"][z]) / fac
            k_eval = np.kron(self.k, fac).reshape(len(self.k), len(self.mu)) / params["alpha_perp"][z]

            # spline interpolation
            k_grid = np.geomspace(np.max([np.min(k_eval), 1e-5]), np.min([np.max(k_eval), 1e3]), 256)
            mu_grid = np.linspace(0., 1., 51)

            ps_idx = 0
            for tr_1, tr_2 in itertools.product(range(self.num_tracers), repeat=2):
                if tr_1 > tr_2: continue

                bias1 = {pname: params['%s_%s_%s' % (pname, tr_1, z)] for pname in self.param_names_bias}
                bias2 = {pname: params['%s_%s_%s' % (pname, tr_2, z)] for pname in self.param_names_bias}

                ctr1 = {pname: params['%s_%s_%s' % (pname, tr_1, z)] for pname in self.param_names_ctr}
                ctr2 = {pname: params['%s_%s_%s' % (pname, tr_2, z)] for pname in self.param_names_ctr}

                # pkmu = self.get_tree_term(k_grid, mu_grid, bias1, bias2, params["fgrowth"][z], params["Dgrowth"][z]) + \
                #        self.get_ctr_terms(k_grid, mu_grid, bias1, bias2, ctr1, ctr2, params["fgrowth"][z], params["Dgrowth"][z]) + \
                #        self.get_stochastic_terms()
                pkmu = self.get_ctr_terms(k_grid, mu_grid, bias1, bias2, ctr1, ctr2, params["fgrowth"][z], params["Dgrowth"][z]) + \
                       self.get_stochastic_terms()

                pkmu *= self.get_damping_factor(k_grid, mu_grid, params["fgrowth"][z])

                # Interpolate to desired k, mu values
                pkmu_interp = RectBivariateSpline(k_grid, mu_grid, pkmu)
                pkmu = pkmu_interp.ev(k_eval, np.tile(mu_eval, (len(self.k), 1)))
                #k_eval = k_eval.reshape(len(self.k), len(mu))
                pkmu = pkmu / (params["alpha_perp"][z]**2 * params["alpha_para"][z])

                # compute the Legendre multipole moments
                pkmu = np.tile(pkmu, (len(self.ells),1,1))
                legendre = np.array([np.tile((2*l+1) * lpmv(0,l,self.mu), (len(self.k),1)) for l in self.ells])
                pk_ell[z, ps_idx] = romb(pkmu * legendre, dx=self.dmu, axis=2)
                ps_idx += 1

        return pk_ell
    
class analytic_tns_terms():
    """"""