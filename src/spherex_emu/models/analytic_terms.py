import numpy as np
from scipy.integrate import romb
from scipy.special import lpmv
from scipy.interpolate import InterpolatedUnivariateSpline, RectBivariateSpline
from scipy.optimize import fsolve
import itertools, math
import symbolic_pofk.linear as linear
from spherex_emu.cosmo_utils import LCDMCosmology, IRResum, get_log_extrap

class analytic_eft_model():
    """Class contatining calculations for the non-emulated terms of the EFT galaxy power spectrum"""

    # default "fiducial" values
    params_cosmo = {'h':0.6736, 'ombh2':0.02218, 'omch2':0.1201, 'As':2.1e-9, 'ns':0.96589, 'fnl':0}
    params_bias = {'galaxy_bias_10':2, 'galaxy_bias_20':0, 'galaxy_bias_G2':0, 'galaxy_bias_Gamma3':0, 'galaxy_bias_01':0, 'galaxy_bias_11':0}
    params_ctr = {'counterterm_0':0, 'counterterm_2':0, 'counterterm_4':0, 'counterterm_fog':0}
    params_stoch = {'P_shot':0, 'a0':0, 'a2':0}
    params_stoch_cross = {'P_shot_cross':0, 'a0_cross':0, 'a2_cross':0}

    def __init__(self, num_tracers, redshift_list, ells, k):

        self.reference_cosmology = LCDMCosmology(0.7, 0.3)

        self.redshift_list = redshift_list
        self.num_zbins = len(redshift_list)
        self.num_tracers = num_tracers
        self.num_spectra = self.num_tracers + math.comb(self.num_tracers, 2)
        self.ells = ells
        self.k = k
        self.mu = np.linspace(0.,1.,2**5+1)
        self.dmu = self.mu[1] - self.mu[0]

        # HACK: hard-coding number density for now
        self.ndens = 0.000501

        self.k_lin = np.geomspace(1e-4, 10., 1000)
        self.sigma_v = 0.
        self.set_default_params()

    def set_default_params(self):
        """sets default cosmology and nuiscance parameters to a dictionary"""
        self.params = {}
        for pname in self.params_cosmo.keys():
            self.params[pname] = self.params_cosmo[pname]
        for (ps, z) in itertools.product(range(self.num_tracers), range(self.num_zbins)):
            for pname in self.params_bias.keys():
                self.params['%s_%s_%s' % (pname, ps, z)] = self.params_bias[pname]
            for pname in self.params_ctr.keys():
                self.params['%s_%s_%s' % (pname, ps, z)] = self.params_ctr[pname]
            for pname in self.params_stoch.keys():
                self.params['%s_%s_%s' % (pname, ps, z)] = self.params_stoch[pname]

    def set_ir_resum_params(self, h, D):
        # object for precalculated IR resummation stuff
        self.irres = IRResum(self.get_pk_lin, hubble=h, rbao=110., kwarg={"D" : D})
        # BAO damping factors
        #t = time.time()
        self.Sigma2 = self.irres.get_Sigma2(ks=0.2)
        self.dSigma2 = self.irres.get_dSigma2(ks=0.2)
        #print("Sigma2 takes {:0.1f}ms to calculate".format((time.time()-t)*1000))

    def set_params(self, param_vector, emu_params_list, analytic_params_list):

        # create a dictionary of parameters for a given (very long) parameter vector. 
        i = 0
        for pname in emu_params_list:
            self.params[pname] = param_vector[i]
            i += 1
        
        for (ps, z) in itertools.product(range(self.num_tracers), range(self.num_zbins)):
            for pname in analytic_params_list:
                self.params['%s_%s_%s' % (pname, ps, z)] = param_vector[i]
                i += 1
        
        # other cosmological parameters
        self.params['Omega_b'] = self.params['ombh2'] / self.params['h']**2
        self.params['Omega_c'] = self.params['omch2'] / self.params['h']**2
        self.params['Omega_m'] = self.params['Omega_b'] + self.params['Omega_c'] # no massive neutrinos assumed in symbolic_pofk
        self.params['sigma8'] = linear.As_to_sigma8(self.params['As'] * 1e9, self.params['Omega_m'], self.params['Omega_b'], self.params['h'], self.params['ns'])
        self.params['k_pivot'] = 0.05 # in unit of 1/Mpc

        # set cosmology
        self.cosmology = LCDMCosmology(self.params["h"], self.params["Omega_m"])

        # linear growth factor & linear growth rate
        self.cosmology.set_Omega_m0(self.params["Omega_m"])
        self.cosmology.set_h(self.params["h"])

        # self._cosmo = LCDMCosmology(params['h'], params['Omega_m'])
        self.params['Dgrowth'] = np.array([self.cosmology.get_Dgrowth(redshift) for redshift in self.redshift_list]) / self.cosmology.get_Dgrowth(0.)
        self.params['fgrowth'] = np.array([self.cosmology.get_fgrowth(redshift) for redshift in self.redshift_list])

        # angular diameter distance & Hubble rate
        DA_list = np.array([self.cosmology.get_D_angular_in_h_inv_Mpc(redshift) for redshift in self.redshift_list])
        Hz_list = np.array([self.cosmology.get_E(redshift) for redshift in self.redshift_list])
        DA_ref_list = np.array([self.reference_cosmology.get_D_angular_in_h_inv_Mpc(redshift) for redshift in self.redshift_list])
        Hz_ref_list = np.array([self.reference_cosmology.get_E(redshift) for redshift in self.redshift_list])
    
        # Alcock-Paczynski effect parameters
        self.params['alpha_perp'] = DA_list / DA_ref_list
        self.params['alpha_para'] = Hz_ref_list / Hz_list

    def calculate_pk_lin(self, k, params):
 
        pk_lin = linear.plin_emulated(k, params['sigma8'], params['Omega_m'], params['Omega_b'], params['h'], params['ns'], emulator='fiducial', extrapolate=True)
        k_extrap, pk_extrap = get_log_extrap(k, pk_lin, xmin=1e-7, xmax=1e7)
        self.pk_lin_spl = InterpolatedUnivariateSpline(np.log(k_extrap), np.log(pk_extrap))

    def get_k_nl(self, D, k0=0.5):
        def func(logk):
            k = np.exp(logk)
            Delta2_lin = k**3 * self.get_pk_lin(k, D) / (2 * np.pi**2)
            return np.log(Delta2_lin)
        
        crit = False
        while crit == False:
            root = fsolve(func, x0=np.log(k0), xtol=1e-6, maxfev=1000) # solve Delta2_lin(k_nl) = 1
            k0 = np.exp(root[0])
            crit = (np.abs(func(np.log(k0))) < 1e-6)

        k_nl = np.exp(root[0])
        return k_nl

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
    
    def get_ctr_terms(self, k, mu, b1_1, b1_2, ctr1, ctr2, f, D):

        plin_tile = self.get_pk_lin_irres_rsd(k, mu, f, D)
        #plin_tile = np.tile(plin, (len(mu),1)).T
        ctr_LO = ( ctr1["counterterm_0"] + ctr2["counterterm_0"])/2. + \
                 ((ctr1["counterterm_2"] + ctr2["counterterm_2"])/2. * f * mu**2) + \
                 ((ctr1["counterterm_4"] + ctr2["counterterm_4"])/2. * f**2 * mu**4)
        ctr_LO = -2 * np.kron(k**2, ctr_LO).reshape(len(k),len(mu)) * plin_tile

        ctr_NLO = (ctr1["counterterm_fog"] + ctr2["counterterm_fog"])/2. * f**4 * mu**4 * (b1_1 + f * mu**2)* (b1_2 + f * mu**2)
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

    def get_stochastic_terms(self, k, mu, stoch1, k_nl, is_cross):
        k = np.atleast_1d(k)
        mu = np.atleast_1d(mu)
        
        # auto power spectrum
        if is_cross == False:
            pkmu = stoch1['P_shot']
            pkmu = pkmu + stoch1['a0'] * np.kron((k / k_nl)**2, lpmv(0,0,mu)).reshape(len(k), len(mu))
            pkmu = pkmu + stoch1['a2'] * np.kron((k / k_nl)**2, lpmv(0,2,mu)).reshape(len(k), len(mu))
            pkmu = 1. / self.ndens * pkmu
        else:
            pkmu = np.zeros((len(k), len(mu)))

        if len(k) == 1 or len(mu) == 1:
            pkmu = np.ravel(pkmu)
        return pkmu

    def get_analytic_terms(self, param_vector, emu_params_list, analytic_params_list):

        if len(param_vector) == len(emu_params_list):
            return 0

        self.set_params(param_vector, emu_params_list, analytic_params_list)

        self.calculate_pk_lin(self.k_lin, self.params)
        self.set_ir_resum_params(self.params["h"], 1.)
        mu_grid = np.linspace(0., 1., 51)

        pk_ell = np.zeros((len(self.redshift_list), self.num_spectra, len(self.ells), len(self.k)))
        for z in range(len(self.redshift_list)):

            # map (k, mu) to different values based on AP effect
            fac = np.sqrt(1 + self.mu**2 * ((self.params["alpha_perp"][z] / self.params["alpha_para"][z])**2 - 1))
            mu_eval = self.mu * (self.params["alpha_perp"][z] / self.params["alpha_para"][z]) / fac
            k_eval = np.kron(self.k, fac).reshape(len(self.k), len(self.mu)) / self.params["alpha_perp"][z]

            # spline interpolation
            k_grid = np.geomspace(np.max([np.min(k_eval), 1e-5]), np.min([np.max(k_eval), 1e3]), 256)
            k_nl = self.get_k_nl(D=self.params["Dgrowth"][z])

            ps_idx = 0
            for tr_1, tr_2 in itertools.product(range(self.num_tracers), repeat=2):
                if tr_1 > tr_2: continue

                b1_1 = self.params[f"galaxy_bias_10_{tr_1}_{z}"]
                b1_2 = self.params[f"galaxy_bias_10_{tr_2}_{z}"]

                ctr1 = {pname: self.params['%s_%s_%s' % (pname, tr_1, z)] for pname in list(self.params_ctr.keys())}
                ctr2 = {pname: self.params['%s_%s_%s' % (pname, tr_2, z)] for pname in list(self.params_ctr.keys())}
                stoch = {pname: self.params['%s_%s_%s' % (pname, tr_1, z)] for pname in list(self.params_stoch.keys())}

                # pkmu = self.get_tree_term(k_grid, mu_grid, bias1, bias2, params["fgrowth"][z], params["Dgrowth"][z]) + \
                #        self.get_ctr_terms(k_grid, mu_grid, bias1, bias2, ctr1, ctr2, params["fgrowth"][z], params["Dgrowth"][z]) + \
                #        self.get_stochastic_terms()
                pkmu = self.get_ctr_terms(k_grid, mu_grid, b1_1, b1_2, ctr1, ctr2, self.params["fgrowth"][z], self.params["Dgrowth"][z]) + \
                       self.get_stochastic_terms(k_grid, mu_grid, stoch, k_nl, tr_1 != tr_2)

                pkmu *= self.get_damping_factor(k_grid, mu_grid, self.params["fgrowth"][z])

                # Interpolate to desired k, mu values
                pkmu_interp = RectBivariateSpline(k_grid, mu_grid, pkmu)
                pkmu = pkmu_interp.ev(k_eval, np.tile(mu_eval, (len(self.k), 1)))
                #k_eval = k_eval.reshape(len(self.k), len(mu))
                pkmu = pkmu / (self.params["alpha_perp"][z]**2 * self.params["alpha_para"][z])

                # compute the Legendre multipole moments
                pkmu = np.tile(pkmu, (len(self.ells),1,1))
                legendre = np.array([np.tile((2*l+1) * lpmv(0,l,self.mu), (len(self.k),1)) for l in self.ells])
                pk_ell[z, ps_idx] = romb(pkmu * legendre, dx=self.dmu, axis=2)
                ps_idx += 1

        return pk_ell.transpose(1,0,3,2) / (self.params["h"])**3
    
class analytic_tns_terms():
    """"""