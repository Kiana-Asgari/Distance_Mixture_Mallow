import numpy as np
import pickle
from scipy.interpolate import RectBivariateSpline
from GMM_diagonalized.sampling import dp_wrapper

def psi_m_wrapper_no_alpha(x, pis, sigma, Delta, alpha, ord, use_lookup_tables=False): 
    psi = psi_m(pis, sigma,alpha, x[0], Delta, use_lookup_tables=use_lookup_tables)
    return np.abs(psi[0]) #np.linalg.norm(psi[0], ord=ord) 


# Wrapper function for multiple workers
def psi_m_wrapper(x, pis, sigma, Delta, ord, use_lookup_tables=False): 
    psi = psi_m(pis, sigma, x[0], x[1], Delta, use_lookup_tables=use_lookup_tables)

    return np.linalg.norm(psi, ord=ord)  


def psi_m(pis: np.ndarray,
          sigma: np.ndarray,
          alpha: float,
          beta:  float,
          Delta: float,
          use_lookup_tables):
    if use_lookup_tables:
        lookup_data = load_lookup_tables(len(sigma), Delta=Delta)
    else:
        lookup_data = None

    pis   = np.asarray(pis,   dtype=np.int16)
    sigma = np.asarray(sigma, dtype=np.int16)

    # helper: d_α and ẟd_α in one pass
    def _d_and_ddiff(diff):
        diff_f  = diff.astype(float)
        diff_a  = diff_f ** alpha
        with np.errstate(divide='ignore'):
            d_dot = diff_a * np.where(diff == 0, 0, np.log(diff_f))
        return diff_a.sum(-1), d_dot.sum(-1)

    # empirical part
    diff_emp = np.abs(pis - sigma)
    d_emp, d_dot_emp = _d_and_ddiff(diff_emp)
    d_emp      = d_emp.mean()
    d_dot_emp  = d_dot_emp.mean()

    # expectation using interpolators
    if use_lookup_tables:
        E_d_interp, E_dot_d_interp = create_interpolators(lookup_data)
        d_mc = E_d_interp(alpha, beta)[0][0]
        d_dot_mc = E_dot_d_interp(alpha, beta)[0][0]
    else:
        Z, E_d, E_dot_d = compute_entry(alpha, beta, len(sigma), Delta)
        d_mc = E_d.item()
        d_dot_mc = E_dot_d.item()
    #print(f'Done for alpha={alpha}, beta={beta}, E - emp={d_mc:.2f}-{d_emp:.2f}, E_dot- emp_dot={d_dot_mc:.2f}-{d_dot_emp:.2f}.')

    # assemble psi
    hat_psi_m = np.array([-d_emp + d_mc, -d_dot_emp + d_dot_mc])

    return hat_psi_m


#################################################
#  Using Lookup Tables for faster computation
#################################################
def load_lookup_tables(n, Delta):
    filename = f'GMM_diagonalized/lookup_tables/mallows_lookup_tables_n{n}_Delta{Delta}.pkl'
    with open(filename, 'rb') as f:
        return pickle.load(f)
    

# Interpolators based on lookup tables
def create_interpolators(lookup_data):
    alpha_vals = lookup_data['alpha_vals']
    try:
        beta_unstandardized_vals = lookup_data['beta_vals']
    except KeyError:
        beta_unstandardized_vals = lookup_data['beta_vals_unstandardized']
    E_d_interp = RectBivariateSpline(alpha_vals, beta_unstandardized_vals, lookup_data['E_d_table'])
    E_dot_d_interp = RectBivariateSpline(alpha_vals, beta_unstandardized_vals, lookup_data['E_dot_d_table'])
    return E_d_interp, E_dot_d_interp


def compute_entry(alpha, beta, n, Delta, h=1e-5):
    #TODO beta = beta_unstandardized/ (n**alpha) # BETA = beta/n^alpha (standardized beta)

    Z = dp_wrapper(n, alpha, beta, Delta)
    perm_alpha_plus = dp_wrapper(n, alpha + h, beta, Delta)
    perm_alpha_minus = dp_wrapper(n, alpha - h, beta, Delta)
    perm_beta_plus = dp_wrapper(n, alpha, beta + h, Delta)
    perm_beta_minus = dp_wrapper(n, alpha, beta - h, Delta)

    E_d = -(np.log(perm_beta_plus) - np.log(perm_beta_minus)) / (2 * h)
    E_dot_d = -(np.log(perm_alpha_plus) - np.log(perm_alpha_minus)) / (2 * h) 
    E_dot_d = E_dot_d/ beta

    return Z, E_d, E_dot_d