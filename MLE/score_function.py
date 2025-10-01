import numpy as np
import pickle
from scipy.interpolate import RectBivariateSpline
import sys
from itertools import permutations

# Wrapper function for multiple workers
def psi_m_wrapper_no_alpha(x, pis, sigma, Delta, alpha, ord=1/2): 
    lookup_data = load_lookup_tables(len(sigma), Delta=Delta)
    psi = psi_m(pis, sigma,alpha, x[0], lookup_data)
    #psi[1] = np.log(np.abs(psi[1]))#/len(sigma) 
    #psi = psi/len(sigma)   # Normalize the second component for more robust optimization

    return np.linalg.norm(psi, ord=2) #+ np.sqrt(len(sigma))*(np.exp(-x[0])) + np.exp(-x[1])  # Return the L1/2 norm of the score
                                         # function for more robust optimization
                                         # when using differential evolution

def psi_m_wrapper(x, pis, sigma, Delta, ord=1/2): 
    lookup_data = load_lookup_tables(len(sigma), Delta=Delta)
    psi = psi_m(pis, sigma, x[0], x[1], lookup_data)
    #psi[1] = np.log(np.abs(psi[1]))#/len(sigma) 
    #psi = psi/len(sigma)   # Normalize the second component for more robust optimization

    return np.linalg.norm(psi, ord=2) #+ np.sqrt(len(sigma))*(np.exp(-x[0])) + np.exp(-x[1])  # Return the L1/2 norm of the score
                                         # function for more robust optimization
                                         # when using differential evolution


# Main function utilizing interpolated lookup tables
def psi_m(pis: np.ndarray,
          sigma: np.ndarray,
          alpha: float,
          beta:  float,
          lookup_data):
    import numpy as np
    from itertools import permutations
    # all the permutations of the first 10 integers



    # temp test
    pis   = np.asarray(pis,   dtype=np.int16)
    sigma = np.asarray(sigma, dtype=np.int16)



    # empirical part

    d_mc, d_dot_mc, alpha_used, beta_used = nearest_neighbor_lookup(alpha, beta, lookup_data)
    alpha = alpha_used
    beta = beta_used


    diff_emp = np.abs(pis - sigma)
    d_emp, d_dot_emp = _d_and_ddiff(diff_emp, alpha)
    d_emp      = d_emp.mean()
    d_dot_emp  = d_dot_emp.mean()
    # variance reduc
    d_emp = np.log(d_emp/len(pis))
    d_dot_emp = np.log(d_dot_emp/len(pis))
    d_mc = np.log(d_mc/len(pis))
    d_dot_mc = np.log(d_dot_mc/len(pis))
    # expectation using interpolators
    # E_d_interp, E_dot_d_interp = create_interpolators(lookup_data)
    # d_mc = E_d_interp(alpha, beta)[0][0]
    # d_dot_mc = E_dot_d_interp(alpha, beta)[0][0]
    # print('for alpha', alpha, 'and beta', beta)
    # print(' empirical: ', f'{d_emp:.1f}', f'{d_dot_emp:.1f}')
    # print(' expected: ', f'{d_mc:.1f}', f'{d_dot_mc:.1f}')
    # assemble psi
    hat_psi_m = np.array([(-d_emp + d_mc), (-d_dot_emp + d_dot_mc)])
    # print('norm 2 psi_m', np.linalg.norm(hat_psi_m, ord=2), '\n')

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

# Nearest-neighbor lookup (returns value at closest grid point + actual alpha/beta used)
def nearest_neighbor_lookup(alpha, beta, lookup_data):
    """
    Find the closest grid point to (alpha, beta) and return the values.
    
    Returns:
        d_mc: E_d value at closest grid point
        d_dot_mc: E_dot_d value at closest grid point
        alpha_used: actual alpha value from grid
        beta_used: actual beta value from grid
    """
    alpha_vals = lookup_data['alpha_vals']
    beta_vals = lookup_data['beta_vals']
    
    # Find closest indices
    alpha_idx = np.argmin(np.abs(alpha_vals - alpha))
    beta_idx = np.argmin(np.abs(beta_vals - beta))
    
    # Get actual grid values
    alpha_used = alpha_vals[alpha_idx]
    beta_used = beta_vals[beta_idx]
    
    # Get table values at closest point
    d_mc = lookup_data['E_d_table'][alpha_idx, beta_idx]
    d_dot_mc = lookup_data['E_dot_d_table'][alpha_idx, beta_idx]
    
    return d_mc, d_dot_mc, alpha_used, beta_used



# helper: d_α and ẟd_α in one pass
def _d_and_ddiff(diff, alpha):
    diff_f  = diff.astype(float)
    diff_a  = diff_f ** alpha
    weights = np.where(diff_a == 0, 1, diff_a)  
    d_dot = diff_a * np.log(weights)
    return diff_a.sum(-1), d_dot.sum(-1)

if __name__ == '__main__':
    import numpy as np
    from itertools import permutations
    pis = np.array([np.arange(100),np.arange(100),np.arange(100)[::-1]])
    sigma = np.arange(100)  
    alpha = 1
    beta = 1
    lookup_data = load_lookup_tables(len(sigma), Delta=7)
    # nearest neighbor lookup
    d_mc, d_dot_mc, alpha_used, beta_used = nearest_neighbor_lookup(alpha, beta, lookup_data)
    alpha = alpha_used
    beta = beta_used
    d_mc = np.log(d_mc) / len(pis)
    d_dot_mc = np.log(d_dot_mc) / len(pis)
    # temp test
    diff_emp_temp= np.abs(pis - sigma)
    d_emp_temp, d_dot_emp_temp = _d_and_ddiff(diff_emp_temp, alpha)
    d_emp_temp      = np.log(d_emp_temp.mean()) / len(pis)
    d_dot_emp_temp  = np.log(d_dot_emp_temp.mean()) / len(pis)
    print(f'for alpha {alpha} and beta {beta}')
    print(f'empirical: d_dot_emp_temp {d_dot_emp_temp:.2f}, d_emp_temp {d_emp_temp:.2f}')

    print(f'expected: d_dot_mc {d_dot_mc:.2f}, d_mc {d_mc:.2f}')

    _d_and_ddiff(diff_emp_temp, alpha)