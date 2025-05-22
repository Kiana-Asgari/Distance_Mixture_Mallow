import numpy as np
import pickle
from scipy.interpolate import RectBivariateSpline


# Wrapper function for multiple workers
def psi_m_wrapper(x, pis, sigma, Delta, ord=1/2): 
    lookup_data = load_lookup_tables(len(sigma), Delta=Delta)
    psi = psi_m(pis, sigma, x[0], x[1], lookup_data)
    psi[1] /= np.log(len(sigma))  # Normalize the second component for more robust optimization

    return np.linalg.norm(psi, ord=ord)  # Return the L1/2 norm of the score
                                         # function for more robust optimization
                                         # when using differential evolution

# Main function utilizing interpolated lookup tables
def psi_m(pis: np.ndarray,
          sigma: np.ndarray,
          alpha: float,
          beta:  float,
          lookup_data):
    
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
    E_d_interp, E_dot_d_interp = create_interpolators(lookup_data)
    d_mc = E_d_interp(alpha, beta)[0][0]
    d_dot_mc = E_dot_d_interp(alpha, beta)[0][0]

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
    beta_vals = lookup_data['beta_vals']
    E_d_interp = RectBivariateSpline(alpha_vals, beta_vals, lookup_data['E_d_table'])
    E_dot_d_interp = RectBivariateSpline(alpha_vals, beta_vals, lookup_data['E_dot_d_table'])
    return E_d_interp, E_dot_d_interp