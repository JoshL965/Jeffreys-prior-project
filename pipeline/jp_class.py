import numpy as np 
import jax 
import jax.numpy as jnp
from typing import NamedTuple, Callable
import pandas as pd
import os
from cosmopower_jax.cosmopower_jax import CosmoPowerJAX as CPJ
from scipy.io import FortranFile
from functools import partial
import scipy
from getdist import plots, MCSamples, loadMCSamples
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)
# ------ GLOBAL QUANTITIES -------
z_grid = jnp.linspace(0, 5.0, 1000)
nbintt = 217
nbinte = 199
nbinee = 199
ellmin=2
plmin_TT = 2
plmin = 30
blmin = np.loadtxt('/cephfs/jlayton/MPhys/data/planck_2018/baseline/plc_3.0/hi_l/plik_lite/plik_lite_v22_TTTEEE.clik/clik/lkl_0/_external/blmin.dat').astype(int)
blmax = np.loadtxt('/cephfs/jlayton/MPhys/data/planck_2018/baseline/plc_3.0/hi_l/plik_lite/plik_lite_v22_TTTEEE.clik/clik/lkl_0/_external/blmax.dat').astype(int)
bin_w = np.loadtxt('/cephfs/jlayton/MPhys/data/planck_2018/baseline/plc_3.0/hi_l/plik_lite/plik_lite_v22_TTTEEE.clik/clik/lkl_0/_external/bweight.dat')
blmin_low_ell = np.loadtxt('/cephfs/jlayton/MPhys/Planck_2018_low_ell/blmin_low_ell.dat').astype(int)
blmax_low_ell = np.loadtxt('/cephfs/jlayton/MPhys/Planck_2018_low_ell/blmax_low_ell.dat').astype(int)
bin_w_low_ell = np.loadtxt('/cephfs/jlayton/MPhys/Planck_2018_low_ell/bweight_low_ell.dat')

bval_low_ell, X_data_low_ell, X_sig_low_ell=np.genfromtxt('/cephfs/jlayton/MPhys/Planck_2018_low_ell/CTT_bin_low_ell_2018.dat', unpack=True)
bval, X_data, X_sig=np.genfromtxt('/cephfs/jlayton/MPhys/data/planck_2018/baseline/plc_3.0/hi_l/plik_lite/plik_lite_v22_TTTEEE.clik/clik/lkl_0/_external/cl_cmb_plik_v22.dat', unpack=True)
blmin_TT=np.concatenate((blmin_low_ell, blmin+len(bin_w_low_ell)))
blmax_TT=np.concatenate((blmax_low_ell, blmax+len(bin_w_low_ell)))
bin_w_TT=np.concatenate((bin_w_low_ell, bin_w))

#EE
lmin_list_EE, lmax_list_EE, mu_LN_EE, sig_LN_EE, loc_LN_EE=np.loadtxt(
        '/cephfs/jlayton/MPhys/data_for_MPhys_extended/Planck_2018_low_ell/lognormal_fit_3bins_EE.txt', unpack=True)
lmin_list_EE=lmin_list_EE.astype('int')
lmax_list_EE=lmax_list_EE.astype('int')

#TT
lmin_list_TT, lmax_list_TT, mu_LN_TT, sig_LN_TT=np.loadtxt(
        '/cephfs/jlayton/MPhys/data_for_MPhys_extended/Planck_2018_low_ell/lognormal_fit_2bins_TT.txt', unpack=True)
lmin_list_TT=lmin_list_TT.astype('int')
lmax_list_TT=lmax_list_TT.astype('int')
ell = np.arange(2, 30)

emulator_cache = {}

# ------ FUNCTIONS FOR LOADING DATA AND THEORY CALCS ---------

def DESI_data(filepath):
    with open(filepath + '/cov_mat.txt') as file:
        list = [[eval(Num) for Num in line.split()] for line in file]
        cov = jnp.array(list)
        cov_inv = jnp.linalg.inv(cov)
    with open(filepath + '/data.txt') as file:
        list = [[eval(Num) for Num in line.split()[0:2]] for line in file.readlines()[1:]]
        array = np.array(list)
        redshifts = array[0:,0]
        data = array[0:,1]
    with open(filepath + '/model_order.txt') as file:
        list = [eval(line) for line in file.readlines()]
        model_order = jnp.array(list)
    return redshifts, data, cov_inv, model_order

def DESI_theory(params, redshift, model_order, extension, cp_DA, cp_H, cp_derived):
    params_array = jnp.array([params[y] for y in param_spaces[extension]])
    derived = cp_derived.predict(params_array)
    if 'hrdrag' in params:
        r_d = params['hrdrag']/params['h']
    else:
        r_d = derived[1]
    c = 299792.458 
    H = cp_H.predict(params_array)
    DA = cp_DA.predict(params_array)
    #DA = jnp.concatenate([jnp.array([0.0]),DA])

    dm = jnp.multiply(jnp.interp(redshift,z_grid[1:],DA), (1+redshift))/r_d
    dh = (c/jnp.interp(redshift,z_grid,H))/r_d
    
    dv = (redshift * dm**2 * dh)**(1.0/3.0)
    
    all_theories = jnp.stack([dm, dh, dv])
    
    return all_theories[model_order, jnp.arange(len(redshift))]

def DES_data(filepath):
    df = pd.read_csv(filepath + '/DES-SN5YR_HD.csv')
    z_cmb = jnp.array(df['zHD'].to_list())
    z_hel = jnp.array(df['zHEL'].to_list())
    dist = np.array(df['MU'].to_list())
    stat = df['MUERR_FINAL'].values
    redshifts = jnp.stack([z_cmb,z_hel])
    with open(filepath + '/covsys_000.txt', 'r') as file:
        array = [eval(line.strip()) for line in file]
        array.pop(0) # This is done as the first line contains N_SN (1829)
        array = np.array(array)
        array = array.reshape((1829,1829))
        C_stat = np.diag(stat**2) 
        cov = array + C_stat
        cov_inv = jnp.linalg.inv(cov)
    
    return redshifts, dist, cov_inv, None

def DES_Doveckie_data(filepath):
    df = pd.read_csv(filepath+'/DES-Dovekie_HD.csv',comment='#',sep=r'\s+')

    z_cmb = jnp.array(df['zHD'].to_list())
    z_hel = jnp.array(df['zHEL'].to_list())
    dist = jnp.array(df['MU'].to_list())
    redshifts = jnp.stack([z_cmb,z_hel])
    with jnp.load(filepath+'/STAT+SYS.npz', 'r') as file:
        cov_upper_triangle = file['cov']
        n = len(z_cmb)
        inv_cov = np.zeros((n,n))
        inv_cov[np.triu_indices(n)] = cov_upper_triangle
        inv_cov_lower = np.tril_indices(n,-1)
        inv_cov[inv_cov_lower] = inv_cov.T[inv_cov_lower]
       
    return redshifts, dist, inv_cov, None

def SN_theory(params, redshift, model_order, extension, cp_DA):
    model_order = 0.0
    M_b = params['M_b']
    params_array = jnp.array([params[y] for y in param_spaces[extension]])
    SN_z_hel = redshift[1]
    SN_z_cmb = redshift[0]
    DA = cp_DA.predict(params_array)
    #DA = jnp.concatenate([jnp.array([0.0]),DA])
    D_L = (1 + SN_z_hel) * (1+ SN_z_cmb) * jnp.interp(SN_z_cmb,z_grid[1:],DA)
    
    Mu_cosmo = (5 * jnp.log10(D_L)) + 25
    
    return Mu_cosmo + M_b

def CMB_theory(params, redshift, model_order, extension, cp_tt, cp_te, cp_ee):
    A_planck = params['A_planck']
    params_array = jnp.array([params[y] for y in param_spaces[extension]])
    ell = cp_tt.modes
    factor = (ell * (ell+1)) / (2*np.pi)
    cl_tt = jnp.divide(cp_tt.predict(params_array) ,factor)

    cl_te = (cp_te.predict(params_array)) / factor
    cl_ee = ((cp_ee.predict(params_array))) / factor   

    Cltt_bin=[]
    for i in range(nbintt):
        cltt=jnp.sum(cl_tt[blmin_TT[i]+plmin_TT-ellmin:blmax_TT[i]+plmin_TT+1-ellmin]*bin_w_TT[blmin_TT[i]:blmax_TT[i]+1])
        Cltt_bin.append(cltt)
    Clte_bin = []
    for i in range(nbinte):
        clte=jnp.sum(cl_te[blmin[i]+plmin-ellmin:blmax[i]+plmin+1-ellmin]*bin_w[blmin[i]:blmax[i]+1])
        Clte_bin.append(clte)
    Clee_bin = []
    for i in range(nbinee):
        clee=jnp.sum(cl_ee[blmin[i]+plmin-ellmin:blmax[i]+plmin+1-ellmin]*bin_w[blmin[i]:blmax[i]+1])
        Clee_bin.append(clee)
    Cl = jnp.concatenate([jnp.array(Cltt_bin),jnp.array(Clte_bin),jnp.array(Clee_bin)])/(A_planck**2)

    return Cl

def CMB_data(filepath):
    cov_cmb_hl = FortranFile(filepath +'/c_matrix_plik_v22.dat', 'r')
    covmat_cmb_hl = cov_cmb_hl.read_reals(dtype=float).reshape((613,613))
    cov_cmb = np.zeros((615,615))
    cov_cmb[0:2, 0:2] = np.diag(X_sig_low_ell**2)
    cov_cmb[2:,2:] = covmat_cmb_hl
    CMB_cov_inv = jnp.linalg.inv(cov_cmb)
    cmb_data=np.concatenate((X_data_low_ell, X_data))

    return None, cmb_data, CMB_cov_inv, None

def PantheonPlus_data(filepath):
    df = pd.read_csv(filepath + '/Pantheon+SH0ES.dat',sep=r'\s+')
    mask = df['zHD'] > 0.01
    df = df[mask]
    z_cmb = jnp.array(df['zHD'].to_list())
    z_hel = jnp.array(df['zHEL'].to_list())
    m_b = jnp.array(df['m_b_corr'].to_list())
    redshifts = jnp.stack([z_cmb,z_hel])
    with open(filepath+'/Pantheon+SH0ES_STAT+SYS.cov','r') as file:
        dim = int(file.readline())
        cov = np.loadtxt(file)[0:]
    cov_matrix = cov.reshape((dim,dim))
    cut_cov_matrix = cov_matrix[np.ix_(mask,mask)]
    inv_cov = jnp.linalg.inv(cut_cov_matrix)

    return redshifts, m_b, inv_cov, None

def Union3_data(filepath):
    df = pd.read_csv(filepath + '/lcparam_full.txt', sep=r'\s+')
    z_cmb = jnp.array(df['zcmb'].to_list())
    z_hel = jnp.array(df['zhel'].to_list())
    m_b = jnp.array(df['mb'].to_list())
    redshifts = jnp.stack([z_cmb,z_hel])
    with open(filepath+ '/mag_covmat.txt') as file:
        dim = int(file.readline())
        cov = np.loadtxt(file)[0:]
    cov_matrix = cov.reshape((dim,dim))
    inv_cov = jnp.linalg.inv(cov_matrix)

    return redshifts, m_b, inv_cov, None

def low_ell_TT_theory(params, redshift, model_order, extension, cp_tt):
    A_planck = params['A_planck']
    params_array = jnp.array([params[y] for y in param_spaces[extension]])
    Dl_tt_low_ell = cp_tt.predict(params_array)[0:30]

    Dl_bin = jnp.array([
        jnp.mean(Dl_tt_low_ell[lmin-2 : lmax-1]) 
        for lmin, lmax in zip(lmin_list_TT, lmax_list_TT)
    ])/(A_planck**2)
    return jnp.log(Dl_bin)

def low_ell_EE_theory(params, redshift, model_order, extension, cp_ee):
    A_planck = params['A_planck']
    params_array = jnp.array([params[y] for y in param_spaces[extension]])
    Dl_ee_low_ell = cp_ee.predict(params_array)[0:30]

    Dl_bin = jnp.array([
        jnp.mean(Dl_ee_low_ell[lmin-2 : lmax-1]) 
        for lmin, lmax in zip(lmin_list_EE, lmax_list_EE)
    ])/(A_planck**2)
    return jnp.log(Dl_bin - loc_LN_EE)

def low_ell_TT_data(filepath):
    inv_cov = jnp.diag(1.0 / sig_LN_TT**2)
    return None, mu_LN_TT, inv_cov, None

def low_ell_EE_data(filepath):
    inv_cov = jnp.diag(1.0 / sig_LN_EE**2)
    return None, mu_LN_EE, inv_cov, None

def high_ell_TTTEEE_data(filepath):
    cov_cmb_hl = FortranFile(filepath +'/c_matrix_plik_v22.dat', 'r')
    covmat_cmb_hl = cov_cmb_hl.read_reals(dtype=float).reshape((613,613))
    CMB_cov_inv = jnp.linalg.inv(covmat_cmb_hl)

    return None, X_data, CMB_cov_inv, None

def high_ell_TTTEEE_theory(params, redshift, model_order, extension, cp_tt,cp_te, cp_ee):
    A_planck = params['A_planck']
    params_array = jnp.array([params[y] for y in param_spaces[extension]])
    ell = cp_tt.modes
    factor = (ell * (ell+1)) / (2*np.pi)
    cl_tt = jnp.divide(cp_tt.predict(params_array) ,factor)

    cl_te = (cp_te.predict(params_array)) / factor
    cl_ee = ((cp_ee.predict(params_array))) / factor   

    Cltt_bin=[]
    for i in range(215):
        cltt=jnp.sum(cl_tt[blmin[i]+plmin-ellmin:blmax[i]+plmin+1-ellmin]*bin_w[blmin[i]:blmax[i]+1])
        Cltt_bin.append(cltt)
    Clte_bin = []
    for i in range(nbinte):
        clte=jnp.sum(cl_te[blmin[i]+plmin-ellmin:blmax[i]+plmin+1-ellmin]*bin_w[blmin[i]:blmax[i]+1])
        Clte_bin.append(clte)
    Clee_bin = []
    for i in range(nbinee):
        clee=jnp.sum(cl_ee[blmin[i]+plmin-ellmin:blmax[i]+plmin+1-ellmin]*bin_w[blmin[i]:blmax[i]+1])
        Clee_bin.append(clee)
    Cl = jnp.concatenate([jnp.array(Cltt_bin),jnp.array(Clte_bin),jnp.array(Clee_bin)])/(A_planck**2)

    return Cl
    
def load_emulators(emulators,extensions):
    active_emulators = []
    for emulator in emulators:
        config = emulator_config[emulator]
        key = f"{emulator}"
        if key not in emulator_cache:
            emulator_cache[key] = CPJ(probe=config['probe'],filepath='/cephfs/jlayton/MPhys/MPhys_extended_emulators/'+extensions+config['filepath'])
        active_emulators.append(emulator_cache[key])
    return active_emulators     

# ----------- GLOBAL DICTIONARIES -----------

catalog = {'BAO_DR1':{'data_filepath':'/cephfs/jlayton/MPhys/data_for_MPhys_extended/DESI_BAO/DR1',
                      'load_fn': DESI_data,
                      'theory_fn': DESI_theory,
                      'emulators': ['cp_DA',
                                    'cp_H',
                                    'cp_derived']},
            'BAO_DR2':{'data_filepath':'/cephfs/jlayton/MPhys/data_for_MPhys_extended/DESI_BAO/DR2',
                      'load_fn': DESI_data,
                      'theory_fn': DESI_theory,
                      'emulators': ['cp_DA',
                                    'cp_H',
                                    'cp_derived']},
            'DES_Y5':{'data_filepath':'/cephfs/jlayton/MPhys/data_for_MPhys_extended/DES_Y5',
                      'load_fn': DES_data,
                      'theory_fn': SN_theory,
                      'emulators': ['cp_DA']},
            'DES_Doveckie':{'data_filepath':'/cephfs/jlayton/MPhys/data_for_MPhys_extended/DES_Doveckie',
                      'load_fn': DES_Doveckie_data,
                      'theory_fn': SN_theory,
                      'emulators': ['cp_DA']},
            'Union3': {'data_filepath': '/cephfs/jlayton/MPhys/data_for_MPhys_extended/Union3',
                       'load_fn': Union3_data,
                       'theory_fn':SN_theory,
                       'emulators':['cp_DA']},
            'CMB_plik_lite':{'data_filepath':'/cephfs/jlayton/MPhys/data_for_MPhys_extended/Planck_2018_plik_lite',
                      'load_fn': CMB_data,
                      'theory_fn': CMB_theory,
                      'emulators': ['cp_tt',
                                    'cp_te',
                                    'cp_ee']},
            'CMB_high_ell_TTTEEE':{'data_filepath':'/cephfs/jlayton/MPhys/data_for_MPhys_extended/Planck_2018_plik_lite',
                      'load_fn': high_ell_TTTEEE_data,
                      'theory_fn': high_ell_TTTEEE_theory,
                      'emulators': ['cp_tt',
                                    'cp_te',
                                    'cp_ee']},
            'CMB_low_ell_TT':{'data_filepath':'/cephfs/jlayton/MPhys/data_for_MPhys_extended/Planck_2018_low_ell',
                      'load_fn': low_ell_TT_data,
                      'theory_fn': low_ell_TT_theory,
                      'emulators': ['cp_tt']},
            'CMB_low_ell_EE':{'data_filepath':'/cephfs/jlayton/MPhys/data_for_MPhys_extended/Planck_2018_low_ell',
                      'load_fn': low_ell_EE_data,
                      'theory_fn': low_ell_EE_theory,
                      'emulators': ['cp_ee']},
            'PantheonPlus': {'data_filepath':'/cephfs/jlayton/MPhys/data_for_MPhys_extended/PantheonPlus',
                      'load_fn': PantheonPlus_data,
                      'theory_fn': SN_theory,
                      'emulators': ['cp_DA']}
           }

emulator_config = {'cp_tt': {'probe':'custom_log',
                            'filepath':'/cmb_tt.npz'},
                    'cp_ee': {'probe': 'custom_log',
                             'filepath': '/cmb_ee.npz'},
                    'cp_te': {'probe': 'custom_pca',
                             'filepath': '/cmb_te.npz'},
                    'cp_H': {'probe': 'custom_log',
                             'filepath': '/background_H.npz'},
                    'cp_DA': {'probe': 'custom_log',
                             'filepath': '/background_Da.npz'},
                    'cp_derived': {'probe': 'custom',
                             'filepath': '/cmb_derived.npz'}
                   }

param_spaces = {'Base': ['ombh2','omch2','h','ns','logA','tau','w0','wa'],
                'Curvature': ['ombh2','omch2','h','ns','logA','tau','w0','wa','omk'],
                'Neutrino_mass': ['ombh2','omch2','h','ns','logA','tau','w0','wa','mnu']}


# class which loads in each data set and assigns it a theory model
class Dataset(NamedTuple):
    x: jax.Array
    y: jax.Array
    inv_covariance: jax.Array
    theory_fn: Callable

class stats_quantities:
    def __init__(self,requested_data, extensions):
    
        datasets_requested = []
        for name in requested_data:
            config = catalog[name]
            data_filepath = config['data_filepath']
            x, y, inv_covariance, model_order = config['load_fn'](data_filepath)
            emulators = load_emulators(config['emulators'],extensions)
            loaded_kwargs = {arg_name: emulator for arg_name, emulator in zip(config['emulators'],emulators)}
            final_theory_fn = partial(config['theory_fn'],**loaded_kwargs,extension=extensions, model_order=model_order if model_order is not None else jnp.array([]))
            dataset = Dataset(x=jnp.array(x) if x is not None else jnp.array([]), y=jnp.array(y), inv_covariance=jnp.array(inv_covariance),theory_fn=final_theory_fn)
            datasets_requested.append(dataset)

        def log_likelihood(params):
            tot_log_like = 0.0
            for data in datasets_requested:
                theory = data.theory_fn(params, data.x)
                log_like = -0.5 * (data.y - theory).T @ data.inv_covariance @ (data.y - theory)
                tot_log_like += log_like
            return tot_log_like 
        
        def chi_square(params):
            tot_chi_squared = 0.0
            for data in datasets_requested:
                theory = data.theory_fn(params, data.x)
                chi2 =  (data.y - theory).T @ data.inv_covariance @ (data.y - theory)
                tot_chi_squared += chi2
            return tot_chi_squared

        def theory(params):
            theory_vec = []
            for data in datasets_requested:
                theory = data.theory_fn(params, data.x)
                theory_vec.append(theory)
            dictionary = {dataset: val for dataset, val in zip(requested_data,theory_vec)}
            return dictionary
        
        def Fisher_matrix(params, sampled_params):
            n = len(sampled_params)
            F_tot = jnp.zeros((n,n))
            for data in datasets_requested:
                grad = jax.jacfwd(data.theory_fn)(params,data.x)
                grad_vector = jnp.array([grad[name] for name in sampled_params]).T
                F = grad_vector.T @ data.inv_covariance @ grad_vector
                F_tot += F
            return F_tot
        
        def Jeffreys_prior(params, sampled_params):
            F_tot = Fisher_matrix(params,sampled_params)
            sign, log_det = jnp.linalg.slogdet(F_tot)
            return 0.5 * log_det
        
        def Inv_cov_matrices():
            inv_cov_matrices = []
            for data in datasets_requested:
                inv_cov = data.inv_covariance
                inv_cov_matrices.append(inv_cov)
            dictionary = {dataset: cov for dataset, cov in zip(requested_data,inv_cov_matrices)}
            return dictionary
        
        def Data():
            data_matrices = []
            for data in datasets_requested:
                vals = data.y
                data_matrices.append(vals)
            dictionary = {dataset: val for dataset, val in zip(requested_data,data_matrices)}
            return dictionary
        
        def Cholesky():
            Cholesky = []
            for data in datasets_requested:
                inv_cov = data.inv_covariance
                Cholesky.append(jnp.linalg.cholesky(jnp.linalg.inv(inv_cov)))
            dictionary = {dataset: cov for dataset, cov in zip(requested_data,Cholesky)}
            return dictionary
        
        def MAP(limits, init_guess, param_names):
            initial_guess = np.array([init_guess[name] for name in param_names])
            limits_ordered = [limits[name] for name in param_names]
            scale_factors = np.where(initial_guess == 0, 1.0, np.abs(initial_guess))
            initial_guess = initial_guess/scale_factors

            scaled_bounds = []
            for b, s in zip(limits_ordered, scale_factors):
                lower = b[0] / s if b[0] is not None else None
                upper = b[1] / s if b[1] is not None else None
                scaled_bounds.append((lower, upper))
            @jax.jit
            def total(params):
                params_dict = {name: val for name,val in zip(param_names,params)}
                tot_chi2 = chi_square(params_dict)
                totals = (0.5*tot_chi2)
                prior_Ap = ((params_dict['A_planck'] - 1.0)**2) / (2 * 0.0025**2)
                totals += prior_Ap
                return jnp.where(jnp.isnan(totals),1e20,totals)
            
            def fun_np(params):
                true_params = params * scale_factors
                return float(total(true_params))
            
            def grad(params):
                true_params = params * scale_factors
                grad_scaled = np.array(jax.grad(total)(true_params)) * scale_factors
                return grad_scaled
            
            res = scipy.optimize.minimize(
                fun=fun_np,
                x0=initial_guess,
                jac=grad,
                method="L-BFGS-B",
                bounds=scaled_bounds,
                options={
                    "maxiter": 60000, 
                    "ftol": 1e-14, 
                    "gtol": 1e-10,
                })
            print(res.success)
            print(res.message)
            
            for name, val in zip(param_names,res.x*scale_factors):
                print(f"{name}:{val:.5f}")

        def getdist_plot(samples_path,plot_path, params_to_plot, param_names,MAP=None,legend_title=None,ncols=None):
            '''
            args
            -----
            samples_path: dictionary with format {getdist legend label: filepath to samples}
            plot_path: path to directory to store getdist plot along with name and format to save as
            params_to_plot: list of parameters to be plotted
            param_names: list of all the parameter names which were sampled
            MAP: Maximum A Posteriori - dictionary with position of MAP.
            '''
            labels = {'w0':r'w_0', 
                      'wa':r'w_a', 
                      'ombh2':r'\Omega_b h^2', 
                      'omch2':r'\Omega_c h^2', 
                      'h':r'h', 
                      'logA':r'\ln(10^{10}A_s)', 
                      'ns':r'n_s', 
                      'tau':r'\tau', 
                      'M_b':r'M_B', 
                      'A_planck':r'A_{\rm Planck}',
                      'omk': r'\Omega_k',
                      'mnu': r'\Sigma m_\nu'
                      }
            plot_list = []
            for name in samples_path:
                samples = np.load(samples_path[name])
                stacked_samples = np.vstack([samples[param] for param in param_names]).T
                GDsamples = MCSamples(samples=stacked_samples,names=param_names,labels=[labels[label]for label in param_names], label=name)
                plot_list.append(GDsamples)
            g = plots.get_subplot_plotter(subplot_size=3,subplot_size_ratio=1)
            g.settings.figure_legend_frame= False
            g.settings.legend_frame = False
            g.settings.lw_contour = 2
            g.settings.linewidth = 2
            g.settings.axes_labelsize = 20
            g.settings.legend_fontsize = 18
            g.settings.tight_layout = True
            

            plt.rcParams['font.family'] = 'serif'
            plt.rcParams['mathtext.fontset'] = 'cm'
            limits = {
                'w0': [-1.3, -0.15],
                'wa': [-2, 0.4]
                }
            g.triangle_plot(
                plot_list,
                params_to_plot,
                filled=True,
                contour_lwd=2,
                legend_labels=[],
                legend_loc=None,
                param_limits=limits
            )

            if MAP is not(None):
                MAP_dict = {label:MAP[label] for label in MAP}
                labels = [s.label for s in plot_list] 
                labels.append('MAP')
            else:
                labels = [s.label for s in plot_list]
                MAP_dict = {}

            g.add_param_markers(MAP_dict,color='purple', ls='--', lw=1.5)
            g.add_param_markers({'w0':-1.0,'wa':0.0},color='grey',ls='--',lw=1)
            labels.append(r'$\Lambda$CDM')
            ax = g.subplots[0,0]
            handles = ax.lines
            g.fig.legend(
                handles,
                labels,
                loc='upper right',
                bbox_to_anchor=(0.99, 0.99),   # X, Y coordinates (1.0 is the edge)
                ncol=1,
                title = legend_title,
                frameon=False,
                fontsize=12,
                title_fontsize=13
            )

            g.export(plot_path)




        self.log_likelihood = log_likelihood
        self.Fisher_matrix = Fisher_matrix
        self.Jeffreys_prior = Jeffreys_prior
        self.theory = theory
        self.inv_cov_matrices = Inv_cov_matrices
        self.data = Data
        self.cholesky = Cholesky
        self.MAP = MAP
        self.chi_square = chi_square
        self.getdist_plot = getdist_plot
        
